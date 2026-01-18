# goal:主要目的是寻找视差区域
# 加载SAM预训练模型
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from config import SAM_CHECKPOINT

sys.path.append("..")


def find_parallax(image, matrix):
    sam_checkpoint = str(SAM_CHECKPOINT)
    model_type = "vit_b"
    device = "cuda"  # "cpu"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    # 通过调用 SamPredictor.set_image 处理图像以生成图像嵌入。 SamPredictor 会记住此嵌入，并将其用于后续掩码预测。
    predictor = SamPredictor(sam)
    # 将输入的图像进行编码
    predictor.set_image(image)
    # Flatten matrix into (x, y) coordinate pairs
    left_parallax1 = np.array(matrix)
    # # Separate x and y coordinates
    x = left_parallax1[:, 0]
    y = left_parallax1[:, 1]
    # # Calculate x mean
    x_mean = np.mean(x)
    # # Calculate y mean
    y_mean = np.mean(y)
    # # 单点 prompt  输入格式为(x, y)和并表示出点所带有的标签1(前景点)或0(背景点)。
    input_point = np.array([[x_mean, y_mean]])  # 标记点
    input_label = np.array([1])  # 点所对应的标签
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    # to show front and back areas
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    # plt.savefig('plt.png')
    # 可以不显示星星
    plt.show()

    # SamPredictor.predict进行分割，模型会返回这些分割目标对应的置信度
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        # 默认给三个
        multimask_output=True,
    )
    # print(masks.shape)  # (number_of_masks) x H x W
    # 画布的大小
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # show_mask(masks[], plt.gca())
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    # get the shadows from show_mask
    mask_image = show_mask(masks[0], plt.gca())
    # cv2.imshow("mask_image1",mask_image1)
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask0, Score: {scores[0]:.3f}", fontsize=6)
    plt.axis('off')
    # plt.savefig('plt1.png')
    plt.show()
    return mask_image

    # for i, (mask, score) in enumerate(zip(masks, scores)):
    #     # 画布的大小
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(image)
    #     show_mask(mask, plt.gca())
    #     show_points(input_point, input_label, plt.gca())
    #     plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=2)
    #     # plt.title(f"Mask {i + 3}, Score: {score:.3f}", fontsize=2)
    #     plt.axis('off')
    #     plt.savefig('plt1.png')
    #     plt.show()

    # fig = plt.figure()
    # img = np.array(fig)
    # img = np.array(img, dtype='uint8')
    # # 现在可以正确显示图像了
    # cv2.imshow("img",img)
    # cv2.imwrite("/home/miao/桌面/img123.png",img)
    # cv2.waitKey()

    # # 多点prompt
    # input_point = np.array([[232.51016, 589.6307], [237.26456, 592.5979], [240.40945, 587.7383]])
    # input_point = np.array(left_parallax1)
    # input_label = np.ones(input_point.shape[0])
    # # input_label = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    #
    # mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    #
    # masks, _, _ = predictor.predict(
    #     point_coords=input_point,
    #     point_labels=input_label,
    #     mask_input=mask_input[None, :, :],
    #     multimask_output=False,
    # )
    #
    # print(masks.shape)
    #
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # show_mask(masks, plt.gca())
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('off')
    # plt.show()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        color = np.array([255 / 255, 255 / 255, 255 / 255, 1])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # mask_image = mask.reshape(h, w) * color.reshape(1, 1, -1)

    ax.imshow(mask_image)
    return mask_image


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


