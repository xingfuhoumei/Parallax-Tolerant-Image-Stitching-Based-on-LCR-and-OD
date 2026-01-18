import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from config import DEFAULT_MAIN_IMAGE, SAM_CHECKPOINT


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


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


def parse_args():
    parser = argparse.ArgumentParser(description="SAM demo for single image segmentation.")
    parser.add_argument("--image", default=DEFAULT_MAIN_IMAGE, help="Path to input image.")
    parser.add_argument("--sam-checkpoint", default=str(SAM_CHECKPOINT), help="Path to SAM checkpoint.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    return parser.parse_args()


args = parse_args()

# 导入待分割图片
image = cv2.imread(args.image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('on')
plt.show()

# 加载SAM预训练模型
import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = args.sam_checkpoint
model_type = "vit_b"

device = args.device  # "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

# 将输入的图像进行编码
predictor.set_image(image)

# 单点 prompt  输入格式为(x, y)和并表示出点所带有的标签1(前景点)或0(背景点)。
input_point = np.array([[270, 240]])  # 标记点
input_label = np.array([1])  # 点所对应的标签

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()

# SamPredictor.predict进行分割，模型会返回这些分割目标对应的置信度
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# /home/miao/桌面/test/(masks.shape)  # (number_of_masks) x H x W

# 三个置信度不同的图
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()

# 多点prompt
# input_point = np.array([[232.51016, 589.6307], [237.26456, 592.5979], [240.40945, 587.7383]])
input_point = np.array([[232.51016,589.6307],
[237.26456 ,592.5979 ],
[240.40945 ,587.7383 ],
[242.6398 ,596.2416],
[242.6398 ,596.2416],
[260.94974 ,587.8843 ],
[262.91092 ,580.1872 ],
[265.7826  ,582.39124],
[266.66684 ,581.60284],
[267.13574 ,577.8806 ],
[268.09396, 537.94336],
[269.82755 ,581.1896 ],
[282.1144  ,586.22406],
[286.04242 ,563.4496 ],
[298.36008 ,534.6031 ],
[300.0464  ,551.77264],
[300.0464  ,551.77264],
[300.36148 ,541.5515 ],
[301.74582 ,537.8154 ],
[181.79973 ,572.0797 ],
[183.3402  ,581.21265],
[191.16719 ,583.9609 ],
[203.95834 ,590.4562 ],
[210.58707 ,581.1869 ],
[217.43652 ,592.53186],
[224.14035 ,581.4999 ],
[224.66289 ,582.5302 ],
[342.53387 ,596.59143],
[170.5739 ,578.972 ],
[229.07278 ,593.18604],
[229.07278 ,593.18604],
[307.35934 ,548.9865 ]])
input_label = np.ones(input_point.shape[0])
# input_label = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
)

print(masks.shape)

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()

# 方框prompt SAM模型可以用一个方框作为输入，格式为[x1,y1,x2,y2],左上，右下。来进行单个目标的分割

# def box_prompt():
#     input_box = np.array([200, 200, 370, 290])
#
#     masks, _, _ = predictor.predict(
#         point_coords=None,
#         point_labels=None,
#         box=input_box[None, :],
#         multimask_output=False,
#     )
#
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     show_mask(masks[0], plt.gca())
#     show_box(input_box, plt.gca())
#     plt.axis('off')
#     plt.show()


# 将点与方框结合，进行目标分割

# def box_point_prompt():
#     input_box = np.array([200, 200, 370, 290])
#     input_point = np.array([[270, 240]])
#     input_label = np.array([1])
#
#     masks, _, _ = predictor.predict(
#         point_coords=input_point,
#         point_labels=input_label,
#         box=input_box,
#         multimask_output=False,
#     )
#
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     show_mask(masks[0], plt.gca())
#     show_box(input_box, plt.gca())
#     show_points(input_point, input_label, plt.gca())
#     plt.axis('off')
#     plt.show()


# 多个方框同时输入，进行多目标分割
# def multi_box_prompt():
#     input_boxes = torch.tensor([
#         [200, 200, 350, 290],
#         [220, 300, 290, 340],
#
#     ], device=predictor.device)
#
#     transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
#     masks, _, _ = predictor.predict_torch(
#         point_coords=None,
#         point_labels=None,
#         boxes=transformed_boxes,
#         multimask_output=False,
#     )
#     print(masks.shape)  # x H x W
#
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     for mask in masks:
#         show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
#     for box in input_boxes:
#         show_box(box.cpu().numpy(), plt.gca())
#     plt.axis('off')
#     plt.show()


# box_prompt()
# box_point_prompt()
# multi_box_prompt()
