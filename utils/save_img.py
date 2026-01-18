import os
import cv2


def SaveImage(img1, img2, path, H4matches, warp_img1, overlap, overlap1, seam_output, H4stitching, matches1=None):
    # 如果没有文件夹自动创造文件夹
    if not os.path.exists(path):
        os.makedirs(path)
    # if matches1 is not None:
    #     cv2.imwrite(path + 'matches1.jpg', matches1)
    # else:
    #     pass
    cv2.imwrite(path + '1-candidate.jpg', img1)
    cv2.imwrite(path + '2-reference.jpg', img2)
    cv2.imwrite(path + '3-ransac_mp.jpg', matches1)
    cv2.imwrite(path + '4-proposed_mp.jpg', H4matches)
    # cv2.imwrite(path + '', H4matches)
    cv2.imwrite(path + '5-overlap_energy.jpg', overlap1)
    cv2.imwrite(path + '6-seam_output.jpg', seam_output)
    cv2.imwrite(path + '7-overlap.jpg', overlap)
    cv2.imwrite(path + '8-H4stitching.jpg', H4stitching)
    print("🚀已成功导入图片进入文件夹")
    return ""


def ShowImage(matches1, H4matches, warp_img1, overlap, overlap1, seam_output, H4stitching):
    if matches1 is not None:
        cv2.imshow("RANSAC_points", matches1)
    else:
        pass
    cv2.imshow('Proposed_stitching', H4matches)
    cv2.imshow('warp_img1', warp_img1)
    cv2.imshow('overlap', overlap)
    cv2.imshow('overlap1', overlap1)
    cv2.imshow('seam_output', seam_output)
    cv2.imshow('H4stitching', H4stitching)
    cv2.waitKey()
    print("图都放完啦,瞅你咋💧")
