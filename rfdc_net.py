import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
import torch.autograd
from natten import NeighborhoodAttention2D as NeighborhoodAttention
from natten import NeighborhoodAttention2D_QKAug as NeighborhoodAttention_QueryAug

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, :out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            # act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  #.flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x


# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, img_size=(128,128)):
        # x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


#########################################
########### feed-forward network #############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x, img_size=(128, 128)):
        # bs x hw x c
        bs, hw, c = x.size()
        # hh = int(math.sqrt(hw))
        hh = img_size[0]
        ww = img_size[1]

        x = self.linear1(x)
        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=ww)
        # bs,hidden_dim,32x32
        x = self.dwconv(x)
        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=ww)
        x = self.linear2(x)
        return x


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = channel
        self.reduction = reduction
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride, groups=out_channels)


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        self.n_feat = n_feat
        self.kernel_size = kernel_size
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v


########### pooling global self-attention #############
class GlobalAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0., pool_kernel=8):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.pool_kernel = pool_kernel

        self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_kernel)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: B, H, W, C
        B, H, W, C = x.shape
        x_down = x.permute(0, 3, 1, 2)  # B, C, H, W
        x_down = self.pool(x_down)
        x_down = x_down.permute(0, 2, 3, 1)  # B, h, w, C
        h, w = H//self.pool_kernel, W//self.pool_kernel
        kv = (
            self.kv(x_down)
            .reshape(B, h*w, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]  # B, num_heads, h*w, head_dim
        q = (
            self.q(x)
            .reshape(B, H*W, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)
        x = x.permute(0, 2, 1, 3).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#########################################
########### RA Module #############
class RALayer(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            kernel_size=11,
            dilation=2,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            token_mlp='leff',
            atten_qaug=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        self.kernel_size = kernel_size
        self.atten_qaug = atten_qaug

        self.norm1 = norm_layer(dim)
        if atten_qaug:
            self.attn = NeighborhoodAttention_QueryAug(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        else:
            self.attn = NeighborhoodAttention(
                dim,
                kernel_size=kernel_size,
                dilation=dilation,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            self.global_attn = GlobalAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            self.region_linear = nn.Linear(dim, 1)
            self.sigmoid = nn.Sigmoid()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        ) if token_mlp == 'ffn' else LeFF(
            dim,
            int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop
        )
        self.CAB = CAB(dim, kernel_size=3, reduction=4, bias=False, act=nn.PReLU())
        # self.w = nn.Parameter(torch.ones(2))

    def forward(self, x, q_aug=None, img_size=(128, 128)):
        # x, q_aug: B, HW, C
        B, L, C = x.shape
        H = img_size[0]
        W = img_size[1]
        assert L == W * H, \
            f"Input image size ({H}*{W} doesn't match model ({L})."

        shortcut = x
        x = self.norm1(x)

        x = x.view(B, H, W, C)

        if not self.atten_qaug:
            with torch.autocast(device_type="cuda", enabled=False):
                x1 = self.attn(x.float())
            x2 = self.global_attn(x)
            local_mask = self.sigmoid(self.region_linear(x))
            global_mask = 1 - local_mask

            x = x1 * local_mask + x2 * global_mask
        else:
            q_aug = q_aug.view(B, H, W, C)
            with torch.autocast(device_type="cuda", enabled=False):
                x = self.attn(x.float(), q_aug.float())

        x = x.view(B, H * W, C)
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=H, w=W)
        x = self.CAB(x)
        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=H, w=W)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), img_size=img_size))
        return x


class DownFRG(nn.Module):
    def __init__(self, dim, n_l_blocks, n_h_blocks, num_heads, mlp_ratio=4,
                        qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                        drop_path=0., norm_layer=nn.LayerNorm, token_mlp='ffn'):
        super().__init__()
        self.dwt = DWT()
        self.l_conv = nn.Conv2d(dim*2, dim, 3, 1, 1)
        self.l_blk = nn.ModuleList([
                RALayer(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, token_mlp=token_mlp, atten_qaug=False)
                for i in range(n_l_blocks)])

        self.h_fusion = nn.Conv2d(dim*3+1, dim, 1, 1, 0)
        self.h_blk = nn.ModuleList([
                RALayer(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, token_mlp=token_mlp, atten_qaug=True,
                        kernel_size=7, dilation=1)
                for i in range(n_h_blocks)])

    def forward(self, x, xi, xm, xb, mask=None):
        x_LL, x_HL, x_LH, x_HH = self.dwt(x)
        b, c, h, w = x_LL.shape
        x_LL = self.l_conv(torch.cat([x_LL, xi], dim=1))
        x_LL = rearrange(x_LL, "b c h w -> b (h w) c").contiguous()
        for l_layer in self.l_blk:
            x_LL = l_layer(x=x_LL, img_size=(h, w))
        x_l_qaug = x_LL
        x_LL = rearrange(x_LL, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        x_h = self.h_fusion(torch.cat([x_HL, x_LH, x_HH, xb], dim=1))
        x_h = rearrange(x_h, "b c h w -> b (h w) c").contiguous()
        for h_layer in self.h_blk:
            x_h = h_layer(x=x_h, q_aug=x_l_qaug, img_size=(h, w))
        x_h = rearrange(x_h, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        return x_LL, x_h


class UpFRG(nn.Module):
    def __init__(self, dim, n_l_blocks, n_h_blocks, num_heads, mlp_ratio=4,
                        qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                        drop_path=0., norm_layer=nn.LayerNorm, token_mlp='ffn'):
        super().__init__()
        self.iwt = IWT()
        self.l_blk = nn.ModuleList([
                RALayer(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, token_mlp=token_mlp, atten_qaug=False)
                for i in range(n_l_blocks)])

        self.h_out_conv = nn.Conv2d(dim, dim * 3, 1, 1, 0)
        self.h_blk = nn.ModuleList([
                RALayer(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, token_mlp=token_mlp, atten_qaug=True,
                        kernel_size=7, dilation=1)
                for i in range(n_h_blocks)])

    def forward(self, x_l, x_h, xm, xb, mask=None):
        b, c, h, w = x_l.shape
        # x_l = self.l_conv(torch.cat([x_l, xm], dim=1))
        x_l = rearrange(x_l, "b c h w -> b (h w) c").contiguous()
        for l_layer in self.l_blk:
            x_l = l_layer(x=x_l, img_size=(h, w))
        x_l_qaug = x_l
        x_l = rearrange(x_l, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        x_h = rearrange(x_h, "b c h w -> b (h w) c").contiguous()
        for h_layer in self.h_blk:
            x_h = h_layer(x=x_h, q_aug=x_l_qaug, img_size=(h, w))
        x_h = rearrange(x_h, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        x_h = self.h_out_conv(x_h)
        x_l = self.iwt(torch.cat([x_l, x_h], dim=1))

        return x_l


class RFDCNet(nn.Module):
    def __init__(self, in_chans=3, embed_dim=64, n_l_blocks=[2,3,3], n_h_blocks=[2,2,2], num_heads=[2, 4, 4],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, token_mlp='leff', **kwargs):
        super(RFDCNet, self).__init__()

        # stochastic depth
        self.num_enc_layers = len(n_l_blocks)
        self.num_dec_layers = len(n_l_blocks)
        depths = [2, 3, 4, 4, 3, 2]
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        dec_dpr = enc_dpr[::-1]

        # Input/Output
        self.input_proj = InputProj(in_channel=3, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=embed_dim, out_channel=in_chans, kernel_size=3, stride=1)

        self.ps_down1 = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d((2**2)*4, embed_dim, 1, 1, 0)
        )
        self.ps_down2 = nn.Sequential(
            nn.PixelUnshuffle(4),
            nn.Conv2d((4**2)*4, embed_dim, 1, 1, 0)
        )
        self.ps_down3 = nn.Sequential(
            nn.PixelUnshuffle(8),
            nn.Conv2d((8**2)*4, embed_dim, 1, 1, 0)
        )

        # encoder of UNet-64
        self.down_group1 = DownFRG(dim=embed_dim, n_l_blocks=n_l_blocks[0], n_h_blocks=n_h_blocks[0], num_heads=num_heads[0],
                                   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                   attn_drop=attn_drop_rate, drop_path=enc_dpr[0:2], norm_layer=norm_layer,
                                   token_mlp=token_mlp)
        self.down_group2 = DownFRG(dim=embed_dim, n_l_blocks=n_l_blocks[1], n_h_blocks=n_h_blocks[1], num_heads=num_heads[1],
                                   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                   attn_drop=attn_drop_rate, drop_path=enc_dpr[2:5], norm_layer=norm_layer,
                                   token_mlp=token_mlp)
        self.down_group3 = DownFRG(dim=embed_dim, n_l_blocks=n_l_blocks[2], n_h_blocks=n_h_blocks[2], num_heads=num_heads[2],
                                   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                   attn_drop=attn_drop_rate, drop_path=enc_dpr[5:9], norm_layer=norm_layer,
                                   token_mlp=token_mlp)

        # decoder of UNet-64
        self.up_group3 = UpFRG(dim=embed_dim, n_l_blocks=n_l_blocks[2], n_h_blocks=n_h_blocks[2], num_heads=num_heads[2],
                                   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                   attn_drop=attn_drop_rate, drop_path=dec_dpr[0:4], norm_layer=norm_layer,
                                   token_mlp=token_mlp)
        self.up_group2 = UpFRG(dim=embed_dim, n_l_blocks=n_l_blocks[1], n_h_blocks=n_h_blocks[1], num_heads=num_heads[1],
                                   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                   attn_drop=attn_drop_rate, drop_path=dec_dpr[4:7], norm_layer=norm_layer,
                                   token_mlp=token_mlp)
        self.up_group1 = UpFRG(dim=embed_dim, n_l_blocks=n_l_blocks[0], n_h_blocks=n_h_blocks[0], num_heads=num_heads[0],
                                   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                   attn_drop=attn_drop_rate, drop_path=dec_dpr[7:9], norm_layer=norm_layer,
                                   token_mlp=token_mlp)

    def forward(self, x, xm, bm):
        # xi = torch.cat((x, xm), dim=1)
        xi = x
        xi_down1 = self.ps_down1(torch.cat([xi, xm], dim=1))
        xi_down2 = self.ps_down2(torch.cat([xi, xm], dim=1))
        xi_down3 = self.ps_down3(torch.cat([xi, xm], dim=1))
        self.img_size = (x.shape[2], x.shape[3])

        xm1 = xm
        xm2 = F.interpolate(xm1, scale_factor=1. / 2, mode='bicubic')
        xm3 = F.interpolate(xm1, scale_factor=1. / 4, mode='bicubic')
        xm4 = F.interpolate(xm1, scale_factor=1. / 8, mode='bicubic')

        bm1 = bm
        bm2 = F.interpolate(bm1, scale_factor=1. / 2, mode='bicubic')
        bm3 = F.interpolate(bm1, scale_factor=1. / 4, mode='bicubic')
        bm4 = F.interpolate(bm1, scale_factor=1. / 8, mode='bicubic')

        ##### shallow conv #####
        x1 = self.input_proj(xi)  # B C H W

        ######## UNet-64 ########
        # Down-path (Encoder)
        x_l, x_H1 = self.down_group1(x1, xi_down1, xm2, bm2)
        x_l, x_H2 = self.down_group2(x_l, xi_down2, xm3, bm3)
        x_l, x_H3 = self.down_group3(x_l, xi_down3, xm4, bm4)

        # Up-path (Decoder)
        x_l = self.up_group3(x_l, x_H3, xm4, bm4)
        x_l = self.up_group2(x_l, x_H2, xm3, bm3)
        x_l = self.up_group1(x_l, x_H1, xm2, bm2)

        ##### Reconstruct #####
        y = self.output_proj(x_l, img_size=self.img_size)

        return y