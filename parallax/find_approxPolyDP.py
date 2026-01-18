# coding: utf-8
import cv2
import numpy as np
import imutils
from PIL import Image
from numba import jit

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# 封装图片显示函数
def image_show(image):
    if image.ndim == 2:
        plt.imshow(image, cmap='gray')
    else:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        plt.imshow(image)
    plt.show()


def find_points(img, x, y):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] != 0:
                if i not in x:
                    x.append(i)
                if j not in y:
                    y.append(j)
    return x, y


def find_area(img_desk):
    x = []
    y = []
    # 读取原图

    # 转换为灰度图
    img_gray = cv.cvtColor(img_desk, cv.COLOR_RGB2GRAY)
    a, b = find_points(img_gray, x, y)
    a = np.array(a)
    b = np.array(b)
    print(min(a), max(a), min(b), max(b))
    new = np.zeros(img_desk.shape)
    for i in range(min(a), max(a) + 1):
        for j in range(min(b), max(b) - 20):
            new[i][j] = [255, 255, 255]
    new = np.array(new, dtype='uint8')
    return new
