import argparse
import numpy
import numpy as np
import math
import cv2
import torch
import pytorch_ssim
from torch.autograd import Variable

from config import DEFAULT_TEST_IMG1, DEFAULT_TEST_IMG2


def parse_args():
    parser = argparse.ArgumentParser(description="Compute PSNR/SSIM between two images.")
    parser.add_argument("--img1", default=DEFAULT_TEST_IMG1, help="Path to first image.")
    parser.add_argument("--img2", default=DEFAULT_TEST_IMG2, help="Path to second image.")
    return parser.parse_args()


args = parse_args()
original = cv2.imread(args.img1)  # numpy.adarray
contrast = cv2.imread(args.img2, 1)


def psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def ssim(img1, img2):
    img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0) / 255.0
    img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0) / 255.0
    img1 = Variable(img1, requires_grad=False)  # torch.Size([256, 256, 3])
    img2 = Variable(img2, requires_grad=False)
    ssim_value = pytorch_ssim.ssim(img1, img2).item()
    return ssim_value


psnrValue = psnr(original, contrast)
ssimValue = ssim(original, contrast)
print(psnrValue)
print(ssimValue)
