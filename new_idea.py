# goal:这个文件主要是实现新的idea:在剔除的外点中得到视差区域.date:2023/05/05
# goal:这个项目是重构了my_stitching这个项目,为了让论文代码可读性更高.date:2023/03/23
import argparse
import time
from pathlib import Path
from matches.Ransac import get_RANSAC_points
from matches.Sift import get_initial_matches
from matches.parallax_area import *
from parallax.SAM import find_parallax
from parallax.find_approxPolyDP import find_area
from stitch.energy_fuction import get_cost
from stitch.graphcut import graph_cut
from stitch.stitching_function import *
from utils.ToPrint import *
from utils.save_img import *
from config import DEFAULT_NEW_IDEA_IMG1, DEFAULT_NEW_IDEA_IMG2, DEFAULT_NEW_IDEA_OUTPUT, ensure_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Run new_idea pipeline.")
    parser.add_argument("--img1", default=DEFAULT_NEW_IDEA_IMG1, help="Path to reference image.")
    parser.add_argument("--img2", default=DEFAULT_NEW_IDEA_IMG2, help="Path to candidate image.")
    parser.add_argument("--output", default=DEFAULT_NEW_IDEA_OUTPUT, help="Output directory.")
    parser.add_argument("--scale", type=float, default=0.5, help="Resize scale for input images.")
    parser.add_argument("--ratio-test", type=float, default=0.98, help="Ratio test threshold.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    start = time.time()
    output_dir = Path(ensure_dir(args.output))
    img1 = cv2.imread(args.img1)
    # cv2.imwrite("/home/gyd/mypaper/test/reference.jpg", img1)
    img2 = cv2.imread(args.img2)
    # cv2.imwrite("/home/gyd/mypaper/test/candidate1.jpg", img2)
    # 图片尺寸比例
    img_scale = args.scale
    img1 = cv2.resize(img1, (int(img_scale * img1.shape[1]), int(img_scale * img1.shape[0])))
    img2 = cv2.resize(img2, (int(img_scale * img2.shape[1]), int(img_scale * img2.shape[0])))
    # ratio_test
    rt = args.ratio_test
    # 希望存储的文件夹 注意打文件夹的括号
    path = str(output_dir)
    # FUNCTION:提取初始匹配对
    src, dst, good, scores = get_initial_matches(img1, img2, rt)
    # FUNCTION:获得我们方法的匹配对
    src_points, dst_points, H, src_after_real_H4, H4matches, left_parallax, right_parallax = get_true_points(img1, img2,
                                                                                                             src, dst,
                                                                                                             good,
                                                                                                             scores)

    stiching_2 = get_stitched_image(img2, img1, H)
    # Function:得到卷曲且裁剪后的candidate图像
    warp_img1, a, b, warp_all = get_warp(img2, img1, H)
    # FUNCTION:利用SAM算法得到视差区域
    image = warp_img1
    if len(left_parallax) != 0:
        left_parallax_shadow = find_parallax(image, left_parallax)
        right_parallax_shadow = find_parallax(img2, right_parallax)
        c = left_parallax_shadow + right_parallax_shadow
        temp_path = output_dir / "parallax_mask.jpg"
        cv2.imwrite(str(temp_path), c)
        c = cv2.imread(str(temp_path))
        flag = True
        new = find_area(c)
    else:
        new = np.zeros(warp_img1.shape)
        flag = False
    # FUNCTION:RANSAC获得匹配对
    src_pts_left, src_after_real_RANSAC_H4, dst_RANSAC_points, matches1, M = get_RANSAC_points(img1, img2, src, dst,
                                                                                               good)
    stiching_ransac = get_stitched_image(img2, img1, M)
    # Function:得到卷曲且裁剪后的candidate图像
    gaussian_img2 = cv2.GaussianBlur(img2, (3, 3), 0.4)
    # 图1重叠区域 ，图2重叠区域
    result_img1, result_img2, m, n, new = find_overlap_region(warp_img1, gaussian_img2, new, flag)

    # FUNCTION:OUR+GRAPHCUT
    # 得到能量函数
    weight, weight1 = get_cost(result_img1, result_img2, dst_points, b, src_after_real_H4, new, flag)
    # 进行graph_cut拼接
    overlap, overlap1, seam_output, H4stitching = graph_cut(result_img1, result_img2, weight, weight1, a, b, m, n, img1,
                                                            img2, H)
    end = time.time()
    # 测试用
    ToPrint(b, path)
    print("总共耗时", end - start)
    # ShowImage(matches1=matches1, H4matches=H4matches, warp_img1=warp_img1, overlap=overlap, overlap1=overlap1,
    #           seam_output=seam_output, H4stitching=H4stitching)
    SaveImage(img1, img2, path, H4matches, warp_img1, overlap, overlap1, seam_output, H4stitching, matches1=matches1)
