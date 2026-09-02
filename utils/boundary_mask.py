import cv2
import os
import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
import torch.nn.functional as F

to_tensor = transforms.ToTensor()

def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            # image_numpy = image_numpy.convert('L')

        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    # image_numpy =
    return np.clip(image_numpy, 0, 255).astype(imtype)


def tensor2gray(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        # if image_numpy.shape[0] == 1:  # grayscale to RGB
        #     # image_numpy = np.tile(image_numpy, (3, 1, 1))
        #     image_numpy = image_numpy.convert('L')

        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    # image_numpy =
    return np.clip(image_numpy, 0, 255).astype(imtype)


def load_img(filepath, mask=False):
    # img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    if mask:
        img = cv2.imread(filepath, 0)
        # img = cv2.resize(img, (512, 512))
        img = img.astype(np.float32)
        # img = np.expand_dims(img, axis=-1).repeat(3, axis=-1)
        img = np.expand_dims(img, axis=-1)
        img = img / 255.
        img = to_tensor(img)

    else:
        img = cv2.imread(filepath)
        # img = cv2.resize(img, (512, 512))
        img = img.astype(np.float32)
        img = img / 255.
        img = to_tensor(img)
    return img


def is_image_file(filename):
    img_extensions = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]
    return any(filename.endswith(extension) for extension in img_extensions)


def dilate(bin_img, ksize=5):
    # 膨胀
    pad = (ksize - 1) // 2
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=pad)
    return out


def erode(bin_img, ksize=5):
    # 腐蚀
    out = 1 - dilate(1 - bin_img, ksize)
    return out


def main():
    path_dir = '/home/nan/dataset/SISR_DATASET/ISTD_adjusted/test/'
    mask_dir = 'test_B'

    mask_files = sorted(os.listdir(os.path.join(path_dir, mask_dir)))
    mask_filenames = [os.path.join(path_dir, mask_dir, x) for x in mask_files if is_image_file(x)]
    mask_img_list = [load_img(x, mask=True) for x in mask_filenames]

    for i in range(len(mask_img_list)):
        mask = mask_img_list[i]
        print(f'{i}, {mask_files[i]}, {mask.shape}')
        save_path = os.path.join(path_dir, 'test_Boundary_box_k7')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        mask = torch.unsqueeze(mask, dim=0)
        # mask = F.interpolate(mask, size=(512, 512), mode='nearest')

        ksize = 7
        mask_dilate = dilate(mask, ksize=ksize)
        mask_erode = erode(mask, ksize=ksize)
        boundary = mask_dilate - mask_erode

        save_img = boundary

        save_img = tensor2gray(save_img)

        cv2.imwrite(os.path.join(save_path, mask_files[i]), save_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])



if __name__ == '__main__':
    main()
