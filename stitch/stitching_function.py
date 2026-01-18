# goal:实现的拼接的功能
import cv2
import numpy as np
from numba import jit




# Function:得到卷曲且裁剪后的candidate图像
def get_warp(img1, img2, M):
    # 获得输入图片的宽、高
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    # 获得图像维度
    img1_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    img2_dims_temp = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

    # 获取第二幅图像的相对透视图
    img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)  # 执行矢量的透视矩阵变换

    # 得到结果图片的维度
    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    # 拼接图像
    # 计算匹配点的维度
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # 仿射变换后创建输出数组
    transform_dist = [-x_min, -y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])

    # 扭曲图像以用于拼接
    result_img = cv2.warpPerspective(img2, transform_array.dot(M),
                                     (x_max - x_min, y_max - y_min))
    # cv2.imshow("result_img",result_img)
    # cv2.waitKey()
    result_img2 = result_img[transform_dist[1]:w1 + transform_dist[1], transform_dist[0]:h1 + transform_dist[0]]
    return result_img2, transform_dist[1], transform_dist[0],result_img

# Function:得到细致的重叠区域
def find_overlap_region(result_img1, result_img2,new,flag):
    a, b = get_w(result_img1)
    resize_img1 = result_img1[:, a:b + 1]
    resize_img2 = result_img2[:, a:b + 1]
    if flag:
        new = new[:, a:b + 1]
        # cv2.imshow("new",new)
    return resize_img1, resize_img2, a, b,new

@jit(nopython=True)
def get_w(wrap_img):
    b = -1
    a = -1
    flag = False
    flag1 = False
    # 以图开始在左侧进行搜索
    # 列数
    for i in range(wrap_img.shape[1]):
        # 行数
        for j in range(wrap_img.shape[0]):
            if wrap_img[j][i][0] == 0 and wrap_img[j][i][1] == 0 and wrap_img[j][i][2] == 0:
                flag = True
            else:
                flag = False
                break
        if flag:
            b = i
            break
    if b != -1 and b != 0:
        return 0, b
    # 图反向遍历
    elif b == 0:
        for i in range(wrap_img.shape[1] - 1, -1, -1):
            for j in range(wrap_img.shape[0]):
                if wrap_img[j][i][0] == 0 and wrap_img[j][i][1] == 0 and wrap_img[j][i][2] == 0:
                    flag1 = True
                else:
                    flag1 = False
                    break
            if flag1:
                a = i
                return a, wrap_img.shape[1] - 1
        # 如果没有找到有效的a值，返回默认值
        return 0, wrap_img.shape[1] - 1
    elif b == -1:
        return 0, wrap_img.shape[1]