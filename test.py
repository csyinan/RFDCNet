import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.loader import get_validation_data
from utils.image_utils import splitimage, mergeimage
import utils

from skimage import img_as_ubyte

from rfdc_net import RFDCNet

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default="/path/to/ISTD_adjusted/test",
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='results/aistd',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='checkpoint/model_aistd.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', default=True, help='Save denoised images in result directory')
parser.add_argument('--cal_metrics', action='store_true', default=False, help='Measure denoised images with GT')
parser.add_argument('--embed_dim', type=int, default=64, help='number of data loading workers')    

parser.add_argument('--ex_name', type=str)


args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = get_validation_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

model_restoration = RFDCNet(embed_dim=args.embed_dim)
print('# model_restoration parameters: %.2f M'%(sum(param.numel() for param in model_restoration.parameters())/ 1e6))

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()

img_multiple_of = 8

with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    rmse_val_rgb = []
    psnr_val_s = []
    ssim_val_s = []
    psnr_val_ns = []
    ssim_val_ns = []
    rmse_val_s = []
    rmse_val_ns = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        torch.cuda.empty_cache()
        test_tile = None
        rgb_gt = data_test[0].numpy().squeeze().transpose((1, 2, 0))
        rgb_noisy = data_test[1].cuda()
        mask = data_test[2].cuda()
        boundary = data_test[3].cuda()
        filenames = data_test[4]

        # For UIUC dataset
        # height, width = rgb_noisy.shape[2], rgb_noisy.shape[3]
        # height, width = 512, 512
        # mask = F.interpolate(mask, size=(height, width), mode='nearest')
        # boundary = F.interpolate(boundary, size=(height, width), mode='nearest')
        # rgb_noisy = F.interpolate(rgb_noisy, size=(height, width), mode='bilinear')

        # Pad the input if not_multiple_of 8
        height, width = rgb_noisy.shape[2], rgb_noisy.shape[3]
        if height >= 1300:
            test_tile = 896
        elif height % img_multiple_of != 0 or width % img_multiple_of != 0:
            H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                        (width + img_multiple_of) // img_multiple_of) * img_multiple_of
            padh = H - height if height % img_multiple_of != 0 else 0
            padw = W - width if width % img_multiple_of != 0 else 0
            rgb_noisy = F.pad(rgb_noisy, (0, padw, 0, padh), 'reflect')
            mask = F.pad(mask, (0, padw, 0, padh), 'reflect')
            boundary = F.pad(boundary, (0, padw, 0, padh), 'reflect')

        if test_tile is None:
            rgb_restored = model_restoration(rgb_noisy, mask, boundary)
        else:
            # test the image tile by tile
            B, C, H, W = rgb_noisy.shape
            tile = min(test_tile, H, W)
            tile_overlap = 100

            split_data, starts = splitimage(rgb_noisy, crop_size=tile, overlap_size=tile_overlap)
            mask_data, starts = splitimage(mask, crop_size=tile, overlap_size=tile_overlap)
            boundary_data, starts = splitimage(boundary, crop_size=tile, overlap_size=tile_overlap)
            for i, (data, mask_, boundary_) in enumerate(zip(split_data, mask_data, boundary_data)):
                split_data[i] = model_restoration(data, mask_, boundary_).cpu()
            rgb_restored = mergeimage(split_data, starts, crop_size=tile, resolution=(B, C, H, W))

        rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))

        # Unpad the output
        if height % img_multiple_of != 0 or width % img_multiple_of != 0:
            rgb_restored = rgb_restored[:height, :width, :]

        if args.save_images:
            utils.save_img(img_as_ubyte(rgb_restored), os.path.join(args.result_dir, filenames[0]))

