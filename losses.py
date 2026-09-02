import torch
import torch.nn as nn
import torch.nn.functional as F


def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class WeightedCharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(WeightedCharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y, weights):
        diff = (x - y) * weights
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
    
from torchvision import models

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.vgg_pretrained_features = models.vgg19(pretrained=True).features
    def forward(self, X, indices=None):
        if indices is None:
            indices = [2, 7, 12, 21, 30]
        out = []
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            if (i + 1) in indices:
                out.append(X)
        return out

class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class VGGLossV2(nn.Module):
    def __init__(self, vgg=None, weights=None):
        super(VGGLossV2, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = weights or [0.1, 0.1, 1, 1, 1]
        self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        self.indices = [2, 7, 12, 21, 30]
    def forward(self, x, y):
        x = x.float()
        x, y = self.normalize(x), self.normalize(y)

        x_vgg = self.vgg(x, self.indices)
        with torch.no_grad():
            y_vgg = self.vgg(y, self.indices)
        # x_vgg, y_vgg = self.vgg(torch.cat([x, y])).chunk(2)
        loss = 0
        for w, fx, fy in zip(self.weights, x_vgg, y_vgg):
            loss += w * self.criterion(fx, fy)
        return loss

class L1_Vgg_losses(nn.Module):
    def __init__(self):
        super(L1_Vgg_losses, self).__init__()
        self.l1 = CharbonnierLoss()
        self.vgg = VGGLossV2()
    def forward(self, x, y):
        return self.l1(x, y) + 0.001 * self.vgg(x, y)


# class L1_Vgg_Freq_losses(nn.Module):
#     def __init__(self):
#         super(L1_Vgg_Freq_losses, self).__init__()
#         self.l1 = CharbonnierLoss()
#         self.vgg = VGGLossV2()
#
#     def forward(self, restored, target, restored_LL, restored_high, target_LL, target_high):
#         return (self.l1(restored, target) + 0.001 * self.vgg(restored, target)
#                 + 0.1 * self.l1(restored_LL, target_LL) + 0.1 * self.l1(restored_high, target_high))


class FreqLoss(nn.Module):
    def __init__(self):
        super(FreqLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, pred, target):
        # with torch.autocast(device_type="cuda", enabled=False):
        #     fft_loss = self.l1_loss(torch.fft.rfft2(pred.float()), torch.fft.rfft2(target.float()))
        with torch.autocast(device_type="cuda", enabled=False):
            target_fft = torch.fft.fft2(target, dim=(-2, -1))
            pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.stack((target_fft.real, target_fft.imag), -1)
        pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
        fft_loss = self.criterion(pred_fft, target_fft)
        return fft_loss


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


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class RegionWaveLoss(nn.Module):
    def __init__(self):
        super(RegionWaveLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()
        self.dwt = DWT()

    def forward(self, pred, target, sm, bm):
        pre_LL1, pred_HL1, pred_LH1, pred_HH1 = self.dwt(pred)
        pre_LL2, pred_HL2, pred_LH2, pred_HH2 = self.dwt(pre_LL1)
        pre_LL3, pred_HL3, pred_LH3, pred_HH3 = self.dwt(pre_LL2)
        target_LL1, target_HL1, target_LH1, target_HH1 = self.dwt(target)
        target_LL2, target_HL2, target_LH2, target_HH2 = self.dwt(target_LL1)
        target_LL3, target_HL3, target_LH3, target_HH3 = self.dwt(target_LL2)

        sm1 = sm  # shadow mask
        sm2 = F.interpolate(sm1, scale_factor=1. / 2, mode='bicubic')
        sm3 = F.interpolate(sm1, scale_factor=1. / 4, mode='bicubic')
        sm4 = F.interpolate(sm1, scale_factor=1. / 8, mode='bicubic')

        bm1 = bm  # boundary mask
        bm2 = F.interpolate(bm1, scale_factor=1. / 2, mode='bicubic')
        bm3 = F.interpolate(bm1, scale_factor=1. / 4, mode='bicubic')
        bm4 = F.interpolate(bm1, scale_factor=1. / 8, mode='bicubic')

        low_loss = (self.criterion(pre_LL1*sm2, target_LL1*sm2) + self.criterion(pre_LL2*sm3, target_LL2*sm3)
                    + self.criterion(pre_LL3*sm4, target_LL3*sm4))
        high_los = (self.criterion(pred_HL1*bm2, target_HL1*bm2) + self.criterion(pred_LH1*bm2, target_LH1*bm2)
                    + self.criterion(pred_HH1*bm2, target_HH1*bm2) + self.criterion(pred_HL2*bm3, target_HL2*bm3)
                    + self.criterion(pred_LH2*bm3, target_LH2*bm3) + self.criterion(pred_HH2*bm3, target_HH2*bm3)
                    + self.criterion(pred_HL3*bm4, target_HL3*bm4) + self.criterion(pred_LH3*bm4, target_LH3*bm4)
                    + self.criterion(pred_HH3*bm4, target_HH3*bm4))
        return low_loss, high_los


class CrossEntropyLoss(nn.Module):
    """Cross Entropy Loss"""

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, input, target):
        # print(input.dtype, target.dtype)
        return self.bce(input, target)
    
class Vgg_loss(nn.Module):
    def __init__(self):
        super(Vgg_loss, self).__init__()
        self.vgg = VGGLossV2()
    def forward(self, x, y):
        return 0.001 * self.vgg(x, y)


if __name__ == '__main__':
    pec_loss = VGGLossV2()
    x = torch.randn(1, 3, 224, 224).cuda()
    y = torch.randn(1, 3, 224, 224).cuda()
    loss = pec_loss(x, y)
    print(f'{loss=}')