def solve_color_difference_with_gain_compensation_and_multiband_blending(img1, img2, H=None, overlap_mask=None):
    """
    使用增益补偿+多频带融合的完整解决方案来解决色差问题
    
    这是按照你的要求实现的方案：
    1. 优先使用增益补偿 (Gain Compensation) 进行全局色彩校正 - foundational step
    2. 配合高质量的多频带融合算法，获得完美的拼接结果
    
    参数:
    - img1: 参考图像 (基准图像)
    - img2: 待校正图像
    - H: 可选的单应性矩阵，如果需要几何变换
    - overlap_mask: 可选的重叠区域mask
    
    返回:
    - 色差校正后的拼接结果
    """
    print("=== 开始色差校正处理 ===")
    print("方案: 增益补偿 (foundational step) + 多频带融合")
    
    # Step 1: 增益补偿 - foundational step for solving color difference
    print("\n步骤1: 应用增益补偿进行全局色彩校正 (foundational step)")
    img2_gain_corrected = gain_compensation(img2, img1, overlap_mask)
    
    # Step 2: 如果需要几何变换
    if H is not None:
        print("\n步骤2: 应用几何变换")
        h, w = img1.shape[:2]
        img2_warped = cv2.warpPerspective(img2_gain_corrected, H, (w, h))
    else:
        img2_warped = img2_gain_corrected
    
    # Step 3: 创建或处理重叠区域mask
    if overlap_mask is None:
        print("\n步骤3: 自动检测重叠区域")
        h, w = img1.shape[:2]
        overlap_mask = np.ones((h, w), dtype=np.uint8) * 255
        # 假设重叠区域在中央部分
        margin_w = w // 6
        overlap_mask[:, :margin_w] = 0
        overlap_mask[:, -margin_w:] = 0
    
    # Step 4: 多频带融合 - 最终获得完美拼接结果
    print("\n步骤4: 应用多频带融合算法获得完美拼接结果")
    final_result = optimized_multiband_blend(img1, img2_warped, overlap_mask, levels=6)
    
    print("\n=== 色差校正完成! ===")
    print("已应用:")
    print("✓ 增益补偿 (Gain Compensation) - foundational step")
    print("✓ 多频带融合 (Multiband Blending)")
    print("✓ 自适应权重计算")
    print("✓ 边缘优化处理")
    
    return final_result


def preprocess_images_for_stitching(img1, img2, method='gain_compensation'):
    """
    拼接前的图像预处理 - 在输入阶段消除色差
    让两张图片色调一致，没有色差
    
    参数:
    - img1: 参考图像 (基准图像)
    - img2: 待校正图像
    - method: 校正方法 ('gain_compensation' 或 'extreme_correction')
    
    返回:
    - img1: 原始参考图像
    - img2_corrected: 色彩校正后的图像
    """
    print("=== 输入图像预处理：消除色差 ===")
    
    if method == 'extreme_correction':
        print("使用极端色彩校正方法...")
        img2_corrected = extreme_color_correction(img2, img1)
    else:
        print("使用增益补偿方法 (foundational step)...")
        img2_corrected = gain_compensation(img2, img1)
    
    # 验证色差改善效果
    original_diff = np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))
    corrected_diff = np.mean(np.abs(img1.astype(np.float32) - img2_corrected.astype(np.float32)))
    
    improvement = ((original_diff - corrected_diff) / original_diff) * 100
    print(f"色差改善效果: {improvement:.1f}% (原始: {original_diff:.1f} -> 校正后: {corrected_diff:.1f})")
    
    if improvement > 10:
        print("✓ 色差显著改善，图像预处理完成")
    else:
        print("! 色差改善有限，可能需要使用极端校正方法")
    
    return img1, img2_corrected


def graph_cut_with_blending(result_img1, result_img2, weight, weight1, a, b, m, n, img1, img2, H):
    """
    改进的图分割拼接函数，集成色彩校正和多频带融合
    解决重叠区域色差问题
    """
    print("🔧 执行增强图分割拼接（带色彩融合）...")
    
    # 步骤1: 传统图分割找接缝
    result_img1_grey = cv2.cvtColor(result_img1, cv2.COLOR_BGR2GRAY)
    result_img2_grey = cv2.cvtColor(result_img2, cv2.COLOR_BGR2GRAY)
    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes(weight.shape)
    
    # function:candidate在右边
    if b == 0:
        # 寻找上下白边
        i, j = get_white_left(weight_find=weight)
        source, sink = (np.inf, 0), (0, np.inf)
        # 右列
        g.add_grid_tedges(nodes[i:j + 1, -1], *sink)  # top right
        # 左
        g.add_grid_tedges(nodes[:-1, 0], *source)  # bottom left
        
        structure = np.array([[0, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0]])
        g.add_grid_edges(nodes, weights=(weight + np.roll(weight, -1, axis=1)), structure=structure,
                         symmetric=True)
        g.add_grid_edges(nodes, weights=(weight + np.roll(weight, -1, axis=0)), structure=structure,
                         symmetric=True)
        max_flow = g.maxflow()
        segments = g.get_grid_segments(nodes)
        H4stitching = get_stitched_image(img2, img1, H)
        
        # 步骤2: 创建融合mask而不是硬拷贝
        blend_mask = create_blend_mask_from_segments(segments, result_img1.shape[:2])
        
        # 步骤3: 多频带融合替代硬拷贝
        print("🌈 应用多频带融合算法...")
        overlap = optimized_multiband_blend(result_img1, result_img2, blend_mask, levels=5)
        
        # 步骤4: 生成接缝可视化
        seam_output = np.zeros(overlap.shape)
        seam = np.logical_or.reduce([np.logical_xor(segments, np.roll(segments, 1, axis=0)),
                                     np.logical_xor(segments, np.roll(segments, 1, axis=1)),
                                     np.logical_xor(segments, np.roll(segments, -1, axis=0)),
                                     np.logical_xor(segments, np.roll(segments, -1, axis=1))])
        seam[:, :1] = seam[:1, :] = seam[:, -1:] = seam[-1:, :] = False
        
        # 显示接缝位置（绿色线条）
        seam_output = overlap.copy()
        seam_output[seam] = [0, 255, 0]
        
        # 有缝的重叠区域可视化
        overlap1 = np.where(seam, [[150]], weight1)
        overlap1 = np.array(overlap1, dtype='uint8')
        
        # 将融合结果放到最终图像中
        seam_output = np.array(seam_output, dtype='uint8')
        w1, h1 = overlap.shape[:2]
        H4stitching[a:w1 + a, m:n + 1] = overlap
        
    # function:candidate在左侧
    else:
        i, j = get_white_right(weight_find=weight)
        source, sink = (np.inf, 0), (0, np.inf)
        # 右列
        g.add_grid_tedges(nodes[0:, -1], *sink)  # top right
        # 左
        g.add_grid_tedges(nodes[i:j + 1, 0], *source)  # bottom left
        
        structure = np.array([[0, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0]])
        g.add_grid_edges(nodes, weights=(weight + np.roll(weight, -1, axis=1)), structure=structure,
                         symmetric=True)
        g.add_grid_edges(nodes, weights=(weight + np.roll(weight, -1, axis=0)), structure=structure,
                         symmetric=True)
        max_flow = g.maxflow()
        segments = g.get_grid_segments(nodes)
        H4stitching = get_stitched_image(img2, img1, H)
        
        # 步骤2: 创建融合mask
        blend_mask = create_blend_mask_from_segments(segments, result_img1.shape[:2])
        
        # 步骤3: 多频带融合（注意这里图像顺序相反）
        print("🌈 应用多频带融合算法...")
        overlap = optimized_multiband_blend(result_img2, result_img1, 255 - blend_mask, levels=5)
        
        # 步骤4: 生成接缝可视化
        seam_output = np.zeros(overlap.shape)
        seam = np.logical_or.reduce([np.logical_xor(segments, np.roll(segments, 1, axis=0)),
                                     np.logical_xor(segments, np.roll(segments, 1, axis=1)),
                                     np.logical_xor(segments, np.roll(segments, -1, axis=0)),
                                     np.logical_xor(segments, np.roll(segments, -1, axis=1))])
        seam[:, :1] = seam[:1, :] = seam[:, -1:] = seam[-1:, :] = False
        
        # 显示接缝位置
        seam_output = overlap.copy()
        seam_output[seam] = [0, 255, 0]
        
        # 将融合结果放到最终图像中
        seam_output = np.array(seam_output, dtype='uint8')
        w1, h1 = overlap.shape[:2]
        H4stitching[a:w1 + a, b:h1 + b] = overlap
        overlap1 = np.where(seam, [[150]], weight1)
        overlap1 = np.array(overlap1, dtype='uint8')
    
    print("✅ 色差校正拼接完成!")
    return overlap, overlap1, seam_output, H4stitching


def create_blend_mask_from_segments(segments, shape):
    """
    从图分割结果创建平滑的融合mask
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 将图分割结果转换为mask（True区域使用img1，False区域使用img2）
    mask[segments] = 255
    
    # 对mask进行平滑处理，创建渐变过渡
    # 1. 距离变换找到边界
    mask_binary = mask > 127
    distance_to_boundary = cv2.distanceTransform(mask_binary.astype(np.uint8), cv2.DIST_L2, 5)
    distance_to_boundary_inv = cv2.distanceTransform((~mask_binary).astype(np.uint8), cv2.DIST_L2, 5)
    
    # 2. 创建平滑的权重过渡（在边界附近创建渐变）
    blend_width = 15  # 融合带宽度
    smooth_mask = np.zeros_like(mask, dtype=np.float32)
    
    # 在接缝附近使用距离变换创建平滑过渡
    total_distance = distance_to_boundary + distance_to_boundary_inv
    
    for i in range(h):
        for j in range(w):
            if total_distance[i, j] < blend_width:
                # 在融合带内，使用平滑权重
                if mask_binary[i, j]:  # 原本属于img1的区域
                    weight = min(1.0, distance_to_boundary_inv[i, j] / blend_width)
                    smooth_mask[i, j] = int(255 * (1 - weight))
                else:  # 原本属于img2的区域
                    weight = min(1.0, distance_to_boundary[i, j] / blend_width)
                    smooth_mask[i, j] = int(255 * weight)
            else:
                # 在融合带外，使用原始mask值
                smooth_mask[i, j] = mask[i, j]
    
    # 3. 高斯平滑进一步减少不连续性
    smooth_mask = cv2.GaussianBlur(smooth_mask, (11, 11), 3.0)
    
    return smooth_mask.astype(np.uint8)


# goal:完成graphcut的拼接
import cv2
import maxflow
import numpy as np
from numba import jit
from PIL import Image


def graph_cut(result_img1, result_img2, weight, weight1, a, b, m, n, img1, img2, H):
    result_img1_grey = cv2.cvtColor(result_img1, cv2.COLOR_BGR2GRAY)
    result_img2_grey = cv2.cvtColor(result_img2, cv2.COLOR_BGR2GRAY)
    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes(weight.shape)
    # function:candidate在右边
    if b == 0:
        # 寻找上下白边
        # fixme:有空看看这个函数,忘记是干什么的了
        i, j = get_white_left(weight_find=weight)
        source, sink = (np.inf, 0), (0, np.inf)
        # 右列
        g.add_grid_tedges(nodes[i:j + 1, -1], *sink)  # top right
        # 下
        # g.add_grid_tedges(nodes[-1, :-1], *source)  # bottom right
        # 左
        g.add_grid_tedges(nodes[:-1, 0], *source)  # bottom left
        # 上
        # g.add_grid_tedges(nodes[0, 1:], *sink)
        structure = np.array([[0, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0]])
        g.add_grid_edges(nodes, weights=(weight + np.roll(weight, -1, axis=1)), structure=structure,
                         symmetric=True)
        # g.add_grid_edges(nodes, weights=(cost + np.roll(cost, -1, axis=0)), structure=np.array([[0, 0, 1]]).T, symmetric=True)
        g.add_grid_edges(nodes, weights=(weight + np.roll(weight, -1, axis=0)), structure=structure,
                         symmetric=True)
        max_flow = g.maxflow()
        segments = g.get_grid_segments(nodes)
        H4stitching = get_stitched_image(img2, img1, H)
        overlap = np.zeros_like(segments, dtype=np.uint8)
        overlap = np.stack([overlap] * 3, axis=2)
        seam_output = np.zeros(overlap.shape)
        seam = np.logical_or.reduce([np.logical_xor(segments, np.roll(segments, 1, axis=0)),
                                     np.logical_xor(segments, np.roll(segments, 1, axis=1)),
                                     np.logical_xor(segments, np.roll(segments, -1, axis=0)),
                                     np.logical_xor(segments, np.roll(segments, -1, axis=1))])
        seam[:, :1] = seam[:1, :] = seam[:, -1:] = seam[-1:, :] = False

        # 拼接好的重叠区域
        overlap, seam_output = combine_overlap1(segments, result_img1, result_img2, overlap, seam, seam_output)
        # 有缝的重叠区域
        overlap1 = np.where(seam, [[150]], weight1)
        overlap1 = np.array(overlap1, dtype='uint8')
        # 获得最终拼接图
        # 矩阵操作后一定要unit8
        seam_output = np.array(seam_output, dtype='uint8')
        w1, h1 = overlap.shape[:2]
        H4stitching[a:w1 + a, m:n + 1] = overlap
    # function:candidate在左侧
    else:
        i, j = get_white_right(weight_find=weight)
        source, sink = (np.inf, 0), (0, np.inf)
        # 右列
        g.add_grid_tedges(nodes[0:, -1], *sink)  # top right
        # 下
        # g.add_grid_tedges(nodes[-1, :-1], *sink)  # bottom right
        # 左
        g.add_grid_tedges(nodes[i:j + 1, 0], *source)  # bottom left
        # 上
        # g.add_grid_tedges(nodes[0, 1:], *source)
        structure = np.array([[0, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0]])
        g.add_grid_edges(nodes, weights=(weight + np.roll(weight, -1, axis=1)), structure=structure,
                         symmetric=True)
        # g.add_grid_edges(nodes, weights=(cost + np.roll(cost, -1, axis=0)), structure=np.array([[0, 0, 1]]).T, symmetric=True)
        g.add_grid_edges(nodes, weights=(weight + np.roll(weight, -1, axis=0)), structure=structure,
                         symmetric=True)
        max_flow = g.maxflow()
        segments = g.get_grid_segments(nodes)
        H4stitching = get_stitched_image(img2, img1, H)
        overlap = np.zeros_like(segments, dtype=np.uint8)
        overlap = np.stack([overlap] * 3, axis=2)
        seam_output = np.zeros(overlap.shape)
        seam = np.logical_or.reduce([np.logical_xor(segments, np.roll(segments, 1, axis=0)),
                                     np.logical_xor(segments, np.roll(segments, 1, axis=1)),
                                     np.logical_xor(segments, np.roll(segments, -1, axis=0)),
                                     np.logical_xor(segments, np.roll(segments, -1, axis=1))])
        seam[:, :1] = seam[:1, :] = seam[:, -1:] = seam[-1:, :] = False
        overlap, seam_output = combine_overlap1(segments, result_img2, result_img1, overlap, seam, seam_output)
        # 矩阵操作后一定要unit8
        seam_output = np.array(seam_output, dtype='uint8')
        w1, h1 = overlap.shape[:2]
        H4stitching[a:w1 + a, b:h1 + b] = overlap
        overlap1 = np.where(seam, [[150]], weight1)
        overlap1 = np.array(overlap1, dtype='uint8')
    return overlap, overlap1, seam_output, H4stitching

# b==0  i是从上往下选 j是从下往上选
@jit(nopython=True)
def get_white_left(weight_find):
    col, row = weight_find.shape
    for i in range(col):
        if weight_find[i, row - 2] > 140:
            continue
        else:
            break
    for j in range(col, -1, -1):
        if weight_find[j - 1, row - 2] > 140:
            continue
        else:
            break
    return i, j

@jit(nopython=True)
def get_white_right(weight_find):
    col, row = weight_find.shape
    for i in range(col):
        if weight_find[i, 1] > 140:
            continue
        else:
            break
    for j in range(col, -1, -1):
        if weight_find[j - 1, 1] > 140:
            continue
        else:
            break
    return i, j


def global_color_matching(source_img, target_img):
    """
    更强力的全局色彩匹配，使用直方图匹配和LAB色彩空间
    """
    # 方法：在LAB色彩空间进行直方图匹配
    source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)
    
    # 对每个LAB通道进行直方图匹配
    matched_lab = np.zeros_like(source_lab)
    
    for channel in range(3):
        source_channel = source_lab[:, :, channel]
        target_channel = target_lab[:, :, channel]
        
        # 计算直方图
        source_hist = cv2.calcHist([source_channel], [0], None, [256], [0, 256])
        target_hist = cv2.calcHist([target_channel], [0], None, [256], [0, 256])
        
        # 计算累积分布函数
        source_cdf = source_hist.cumsum()
        target_cdf = target_hist.cumsum()
        
        # 标准化CDF
        source_cdf_normalized = source_cdf / source_cdf[-1]
        target_cdf_normalized = target_cdf / target_cdf[-1]
        
        # 创建查找表进行映射
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            # 找到最接近的目标值
            diff = np.abs(target_cdf_normalized - source_cdf_normalized[i])
            lut[i] = np.argmin(diff)
        
        # 应用查找表
        matched_lab[:, :, channel] = cv2.LUT(source_channel, lut)
    
    # 转换回BGR
    result = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)
    
    # 额外的色温调整 - 如果还有明显差异
    result_float = result.astype(np.float32)
    target_float = target_img.astype(np.float32)
    
    # 计算整体色温差异并微调
    result_mean = np.mean(result_float, axis=(0, 1))
    target_mean = np.mean(target_float, axis=(0, 1))
    
    # 轻微调整以匹配整体色调
    adjustment = (target_mean - result_mean) * 0.3  # 30%的调整强度
    result_float += adjustment
    
    return np.clip(result_float, 0, 255).astype(np.uint8)


def get_stitched_image(img1, img2, M):
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
    result_img2 = result_img[transform_dist[1]:w1 + transform_dist[1], transform_dist[0]:h1 + transform_dist[0]]

    # 先将第一张图像放到结果图像中
    result_img[transform_dist[1]:w1 + transform_dist[1], transform_dist[0]:h1 + transform_dist[0]] = img1

    return np.array(result_img)


def simple_blend_preprocessed_images(img1, img2, mask):
    """
    用于预处理过的图像的简化融合方法
    由于色差已经在预处理阶段解决，只需要简单的几何融合
    """
    result = img1.copy()
    overlap_mask = mask > 0
    
    if not np.any(overlap_mask):
        return result
    
    # 由于色差已经校正，使用简单的渐变融合
    h, w = mask.shape
    
    # 找到重叠区域的边界
    coords = np.where(overlap_mask)
    if len(coords[1]) > 0:
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        
        # 创建从左到右的渐变权重
        for x in range(x_min, x_max + 1):
            if x_max > x_min:
                weight = (x - x_min) / (x_max - x_min)
            else:
                weight = 0.5
            
            # 平滑的S型权重曲线
            weight = 3 * weight**2 - 2 * weight**3
            
            # 应用权重融合
            col_mask = overlap_mask[:, x]
            if np.any(col_mask):
                result[col_mask, x] = (
                    img1[col_mask, x] * (1 - weight) + 
                    img2[col_mask, x] * weight
                ).astype(np.uint8)
    
    return result


def simple_color_correction_blend(img1, img2, mask):
    """
    最简单的融合方法，因为已经在早期做了色彩匹配
    """
    result = img1.copy()
    overlap_mask = mask > 0
    
    if not np.any(overlap_mask):
        return result
    
    # 简单的50:50混合
    img1_pil = Image.fromarray(img1)
    img2_pil = Image.fromarray(img2)
    blended = Image.blend(img1_pil, img2_pil, 0.5)
    blended_array = np.array(blended)
    
    # 只在重叠区域应用混合结果
    result[overlap_mask] = blended_array[overlap_mask]
    
    return result


def advanced_multiband_blending(img1, img2, mask, levels=6):
    """
    先进的多频带混合算法
    基于 Burt & Adelson "A multiresolution spline with application to image mosaics" 论文
    结合现代改进技术
    """
    h, w = img1.shape[:2]
    
    # 1. 预处理：增益补偿 (foundational step)
    img2_corrected = gain_compensation(img2, img1, mask)
    
    # 2. 创建优化的融合mask
    blend_mask = create_optimal_blend_mask(mask, img1, img2_corrected)
    
    # 3. 多频带拉普拉斯金字塔融合
    result = multiband_pyramid_blend(img1, img2_corrected, blend_mask, levels)
    
    # 4. 后处理：梯度域优化
    result = gradient_domain_optimization(result, img1, img2_corrected, mask)
    
    return result


def optimized_multiband_blend(img1, img2, mask=None, levels=5):
    """
    优化的多频带融合算法 - 专门用于解决色差问题
    结合增益补偿和多频带技术的最佳实践
    增强版本，对色差更敏感
    """
    h, w = img1.shape[:2]
    
    # 如果没有提供mask，创建重叠区域mask
    if mask is None:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        # 假设重叠区域在图像中间部分
        margin = w // 4
        mask[:, :margin] = 0
        mask[:, -margin:] = 0
    
    # Step 1: 再次强化增益补偿 (foundational step)
    print("应用强化增益补偿进行全局色彩校正...")
    img2_gain_corrected = enhanced_gain_compensation(img2, img1, mask)
    
    # Step 2: 额外的直方图匹配
    img2_hist_matched = histogram_matching_lab(img2_gain_corrected, img1, mask)
    
    # Step 3: 创建更智能的融合权重
    blend_weights = create_color_aware_blend_weights(mask, img1, img2_hist_matched)
    
    # Step 4: 多频带拉普拉斯金字塔融合
    print("执行增强多频带融合...")
    result = enhanced_multiband_pyramid_blend(img1, img2_hist_matched, blend_weights, levels)
    
    # Step 5: 后处理优化
    result = post_process_blend(result, img1, img2_hist_matched, mask)
    
    return result


def enhanced_gain_compensation(img_source, img_target, mask):
    """
    增强的增益补偿，专门处理严重色差
    """
    # 先应用基础增益补偿
    img_corrected = gain_compensation(img_source, img_target, mask)
    
    # 再进行迭代优化
    valid_mask = mask > 0
    if not np.any(valid_mask):
        return img_corrected
    
    # 计算残余色差
    corrected_mean = np.mean(img_corrected[valid_mask], axis=0)
    target_mean = np.mean(img_target[valid_mask], axis=0)
    
    # 如果色差仍然很大，进行二次校正
    color_error = np.linalg.norm(corrected_mean - target_mean)
    if color_error > 20:  # 色差阈值
        print(f"检测到残余色差 {color_error:.1f}，应用二次校正...")
        
        # 计算额外的色彩调整
        extra_adjustment = (target_mean - corrected_mean) * 0.5
        
        img_corrected_float = img_corrected.astype(np.float32)
        img_corrected_float += extra_adjustment
        img_corrected = np.clip(img_corrected_float, 0, 255).astype(np.uint8)
    
    return img_corrected


def histogram_matching_lab(img_source, img_target, mask):
    """
    在LAB色彩空间进行精确的直方图匹配
    """
    valid_mask = mask > 0
    if not np.any(valid_mask):
        return img_source
    
    # 转换到LAB色彩空间
    source_lab = cv2.cvtColor(img_source, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(img_target, cv2.COLOR_BGR2LAB)
    
    result_lab = source_lab.copy()
    
    # 只对重叠区域进行直方图匹配
    for channel in range(3):
        source_channel = source_lab[:, :, channel]
        target_channel = target_lab[:, :, channel]
        
        # 提取重叠区域的直方图
        source_overlap = source_channel[valid_mask]
        target_overlap = target_channel[valid_mask]
        
        # 计算直方图
        if channel == 0:  # L通道
            hist_range = [0, 100]
            bins = 100
        else:  # A和B通道
            hist_range = [0, 255]
            bins = 256
        
        source_hist = np.histogram(source_overlap, bins=bins, range=hist_range)[0]
        target_hist = np.histogram(target_overlap, bins=bins, range=hist_range)[0]
        
        # 计算累积分布函数
        source_cdf = np.cumsum(source_hist).astype(np.float64)
        target_cdf = np.cumsum(target_hist).astype(np.float64)
        
        # 标准化CDF
        if source_cdf[-1] > 0:
            source_cdf /= source_cdf[-1]
        if target_cdf[-1] > 0:
            target_cdf /= target_cdf[-1]
        
        # 创建查找表
        lut = np.zeros(bins, dtype=np.uint8)
        for i in range(bins):
            # 找到最接近的目标值
            if source_cdf[i] > 0:
                diff = np.abs(target_cdf - source_cdf[i])
                lut[i] = np.argmin(diff)
            else:
                lut[i] = i
        
        # 应用查找表（只在有效范围内）
        if channel == 0:  # L通道
            valid_indices = (source_channel >= 0) & (source_channel < 100)
            mapped_values = np.interp(source_channel[valid_indices], 
                                    np.arange(100), lut[:100])
        else:  # A和B通道
            valid_indices = (source_channel >= 0) & (source_channel < 256)
            mapped_values = lut[source_channel[valid_indices]]
        
        result_lab[valid_indices, channel] = mapped_values
    
    # 转换回BGR
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    return result_bgr


def create_color_aware_blend_weights(mask, img1, img2):
    """
    创建对色差敏感的自适应融合权重
    """
    h, w = mask.shape
    weights = np.zeros((h, w), dtype=np.float32)
    valid_region = mask > 0
    
    if not np.any(valid_region):
        return weights
    
    # 转换到LAB空间计算色差
    img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB).astype(np.float32)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # 计算像素级色差
    color_diff = np.linalg.norm(img1_lab - img2_lab, axis=2)
    
    # 计算图像质量指标
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 计算局部对比度
    kernel = np.ones((5, 5), np.float32) / 25
    contrast1 = cv2.filter2D(img1_gray.astype(np.float32), -1, 
                           cv2.getGaussianKernel(5, 1.0) @ cv2.getGaussianKernel(5, 1.0).T)
    contrast2 = cv2.filter2D(img2_gray.astype(np.float32), -1,
                           cv2.getGaussianKernel(5, 1.0) @ cv2.getGaussianKernel(5, 1.0).T)
    
    # 计算距离变换
    distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # 找到重叠区域边界
    y_coords, x_coords = np.where(valid_region)
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    
    # 为每个像素计算智能权重
    for y, x in zip(y_coords, x_coords):
        # 1. 基于位置的渐变权重
        if x_max > x_min:
            position_weight = (x - x_min) / (x_max - x_min)
        else:
            position_weight = 0.5
        
        # 2. 基于色差的权重 (色差小的地方权重更均匀)
        local_color_diff = color_diff[y, x]
        if local_color_diff > 30:  # 色差大的区域
            color_weight = 0.5  # 使用平均权重
        else:  # 色差小的区域
            color_weight = position_weight  # 使用位置权重
        
        # 3. 基于图像质量的权重
        quality_weight = 0.5
        if contrast1[y, x] > contrast2[y, x] * 1.2:
            quality_weight = 0.3  # 偏向img1
        elif contrast2[y, x] > contrast1[y, x] * 1.2:
            quality_weight = 0.7  # 偏向img2
        
        # 4. 基于距离的稳定性权重
        distance_weight = min(distance_transform[y, x] / 15.0, 1.0)
        
        # 综合权重计算，色差权重占主导
        combined_weight = (
            0.4 * color_weight +               # 40% 色差相关权重
            0.3 * position_weight +            # 30% 位置权重
            0.2 * quality_weight +             # 20% 质量权重
            0.1 * distance_weight              # 10% 距离权重
        )
        
        # 使用更平滑的S型曲线
        smooth_weight = 6 * combined_weight**5 - 15 * combined_weight**4 + 10 * combined_weight**3
        weights[y, x] = np.clip(smooth_weight, 0.05, 0.95)
    
    # 更强的高斯平滑
    weights = cv2.GaussianBlur(weights, (31, 31), 10.0)
    
    return weights


def extreme_color_correction(img_source, img_target, overlap_mask=None):
    """
    极端色差校正算法 - 专门处理严重的色彩不匹配
    结合多种色彩校正技术的强化版本
    """
    if overlap_mask is None:
        h, w = img_source.shape[:2]
        overlap_mask = np.ones((h, w), dtype=np.uint8) * 255
    
    valid_mask = overlap_mask > 0
    if not np.any(valid_mask):
        return img_source
    
    # 步骤1: 在LAB空间进行精确的统计匹配
    img_source_lab = cv2.cvtColor(img_source, cv2.COLOR_BGR2LAB).astype(np.float64)
    img_target_lab = cv2.cvtColor(img_target, cv2.COLOR_BGR2LAB).astype(np.float64)
    
    source_overlap = img_source_lab[valid_mask]
    target_overlap = img_target_lab[valid_mask]
    
    result_lab = img_source_lab.copy()
    
    # 对每个LAB通道进行精确匹配
    for channel in range(3):
        source_ch = source_overlap[:, channel]
        target_ch = target_overlap[:, channel]
        
        # 使用分位数回归进行鲁棒匹配
        source_percentiles = np.percentile(source_ch, [5, 25, 50, 75, 95])
        target_percentiles = np.percentile(target_ch, [5, 25, 50, 75, 95])
        
        # 分段线性变换
        result_channel = result_lab[:, :, channel].copy()
        
        # 映射每个分段
        for i in range(len(source_percentiles) - 1):
            src_low, src_high = source_percentiles[i], source_percentiles[i + 1]
            tgt_low, tgt_high = target_percentiles[i], target_percentiles[i + 1]
            
            # 找到在当前分段的像素
            mask_segment = (result_channel >= src_low) & (result_channel <= src_high)
            
            if np.any(mask_segment) and (src_high - src_low) > 1e-6:
                # 线性插值映射
                ratio = (result_channel[mask_segment] - src_low) / (src_high - src_low)
                result_channel[mask_segment] = tgt_low + ratio * (tgt_high - tgt_low)
        
        result_lab[:, :, channel] = result_channel
    
    # 确保LAB值在有效范围内
    result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 100)
    result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], -127, 127)
    result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], -127, 127)
    
    # 转换回BGR
    result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # 步骤2: 在HSV空间进行色相和饱和度微调
    result_hsv = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    target_hsv = cv2.cvtColor(img_target, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # 调整色相和饱和度
    result_mean_hsv = np.mean(result_hsv[valid_mask], axis=0)
    target_mean_hsv = np.mean(target_hsv[valid_mask], axis=0)
    
    # 色相调整（循环处理）
    hue_diff = target_mean_hsv[0] - result_mean_hsv[0]
    if hue_diff > 90:
        hue_diff -= 180
    elif hue_diff < -90:
        hue_diff += 180
    
    result_hsv[:, :, 0] = (result_hsv[:, :, 0] + hue_diff * 0.3) % 180
    
    # 饱和度调整
    sat_ratio = target_mean_hsv[1] / (result_mean_hsv[1] + 1e-6)
    sat_ratio = np.clip(sat_ratio, 0.5, 2.0)
    result_hsv[:, :, 1] *= sat_ratio * 0.7 + 0.3  # 部分调整
    
    # 转换回BGR
    result_hsv = np.clip(result_hsv, 0, 255)
    result_final = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # 步骤3: 最终的RGB微调
    result_float = result_final.astype(np.float64)
    target_float = img_target.astype(np.float64)
    
    # 计算整体均值调整
    result_mean_final = np.mean(result_float[valid_mask], axis=0)
    target_mean_final = np.mean(target_float[valid_mask], axis=0)
    
    final_adjustment = (target_mean_final - result_mean_final) * 0.2
    result_float += final_adjustment
    
    return np.clip(result_float, 0, 255).astype(np.uint8)


def solve_extreme_color_difference(img1, img2, H=None, overlap_mask=None):
    """
    解决极端色差问题的完整方案
    使用最强的色彩校正 + 优化的多频带融合
    """
    print("=== 极端色差校正方案 ===")
    print("使用: 极端色彩校正 + 增强多频带融合")
    
    # Step 1: 极端色彩校正 (foundational step)
    print("\n步骤1: 应用极端色彩校正 (foundational step)")
    img2_extreme_corrected = extreme_color_correction(img2, img1, overlap_mask)
    
    # Step 2: 几何变换（如需要）
    if H is not None:
        print("\n步骤2: 应用几何变换")
        h, w = img1.shape[:2]
        img2_warped = cv2.warpPerspective(img2_extreme_corrected, H, (w, h))
    else:
        img2_warped = img2_extreme_corrected
    
    # Step 3: 处理重叠区域mask
    if overlap_mask is None:
        print("\n步骤3: 自动检测重叠区域")
        h, w = img1.shape[:2]
        overlap_mask = np.ones((h, w), dtype=np.uint8) * 255
        margin_w = w // 8  # 更大的重叠区域
        overlap_mask[:, :margin_w] = 0
        overlap_mask[:, -margin_w:] = 0
    
    # Step 4: 增强多频带融合
    print("\n步骤4: 应用增强多频带融合")
    
    # 创建极端色差感知的权重
    weights = create_extreme_color_aware_weights(overlap_mask, img1, img2_warped)
    
    # 多频带融合
    final_result = enhanced_multiband_pyramid_blend(img1, img2_warped, weights, levels=6)
    
    # 最终后处理
    final_result = extreme_post_process(final_result, img1, img2_warped, overlap_mask)
    
    print("\n=== 极端色差校正完成! ===")
    return final_result


def create_extreme_color_aware_weights(mask, img1, img2):
    """
    为极端色差情况创建特殊的融合权重
    """
    h, w = mask.shape
    weights = np.zeros((h, w), dtype=np.float32)
    valid_region = mask > 0
    
    if not np.any(valid_region):
        return weights
    
    # 计算像素级色差 (在LAB空间)
    img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB).astype(np.float32)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    color_diff = np.linalg.norm(img1_lab - img2_lab, axis=2)
    
    # 计算局部纹理复杂度
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 使用拉普拉斯算子计算纹理
    texture1 = cv2.Laplacian(img1_gray, cv2.CV_64F)
    texture2 = cv2.Laplacian(img2_gray, cv2.CV_64F)
    texture1 = np.abs(texture1)
    texture2 = np.abs(texture2)
    
    # 距离变换
    distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    y_coords, x_coords = np.where(valid_region)
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    
    for y, x in zip(y_coords, x_coords):
        # 基于位置的基础权重
        if x_max > x_min:
            position_weight = (x - x_min) / (x_max - x_min)
        else:
            position_weight = 0.5
        
        # 色差越大，越倾向于使用平均权重
        local_color_diff = color_diff[y, x]
        if local_color_diff > 50:  # 极端色差
            color_factor = 0.5  # 强制平均
        elif local_color_diff > 25:  # 中等色差
            color_factor = 0.3 + 0.4 * position_weight  # 部分平均
        else:  # 小色差
            color_factor = position_weight  # 正常位置权重
        
        # 纹理质量权重
        texture_factor = 0.5
        if texture1[y, x] > texture2[y, x] * 1.5:
            texture_factor = 0.2  # 偏向img1
        elif texture2[y, x] > texture1[y, x] * 1.5:
            texture_factor = 0.8  # 偏向img2
        
        # 距离权重（边界附近更保守）
        distance_factor = distance_transform[y, x] / 20.0
        distance_factor = min(distance_factor, 1.0)
        
        # 综合权重，色差因子占主导
        combined_weight = (
            0.6 * color_factor +        # 60% 色差因子
            0.2 * texture_factor +      # 20% 纹理因子
            0.2 * distance_factor       # 20% 距离因子
        )
        
        # 更激进的S型曲线，增强极端值
        if combined_weight < 0.3:
            smooth_weight = 2 * combined_weight**2
        elif combined_weight > 0.7:
            smooth_weight = 1 - 2 * (1 - combined_weight)**2
        else:
            smooth_weight = combined_weight
        
        weights[y, x] = np.clip(smooth_weight, 0.1, 0.9)
    
    # 强平滑以确保自然过渡
    weights = cv2.GaussianBlur(weights, (41, 41), 15.0)
    
    return weights


def extreme_post_process(img, img1, img2, mask):
    """
    极端色差情况的后处理
    """
    result = img.copy().astype(np.float32)
    
    # 检测并修复色彩不连续性
    valid_mask = mask > 0
    
    if np.any(valid_mask):
        # 计算整体色彩一致性
        result_mean = np.mean(result[valid_mask], axis=0)
        img1_mean = np.mean(img1[valid_mask], axis=0)
        
        color_consistency_error = np.linalg.norm(result_mean - img1_mean)
        
        if color_consistency_error > 15:
            print(f"应用色彩一致性修正 (误差: {color_consistency_error:.1f})")
            # 全局色彩校正
            global_correction = (img1_mean - result_mean) * 0.3
            result += global_correction
    
    # 边界区域特殊处理
    boundary_mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((21,21), np.uint8))
    boundary_mask = boundary_mask > 0
    
    if np.any(boundary_mask):
        # 多尺度边界平滑
        for channel in range(3):
            boundary_region = result[:,:,channel]
            
            # 三级平滑
            smoothed1 = cv2.bilateralFilter(boundary_region.astype(np.uint8), 13, 80, 80)
            smoothed2 = cv2.bilateralFilter(smoothed1, 9, 60, 60)
            smoothed3 = cv2.bilateralFilter(smoothed2, 5, 40, 40)
            
            # 渐进式混合
            result[boundary_mask, channel] = (
                result[boundary_mask, channel] * 0.5 + 
                smoothed3[boundary_mask] * 0.5
            )
    
    return np.clip(result, 0, 255).astype(np.uint8)


def fine_tune_color_balance(img_corrected, img_reference, mask):
    """
    在增益补偿基础上进行精细的色彩平衡调整
    """
    valid_mask = mask > 0
    if not np.any(valid_mask):
        return img_corrected
    
    img_corrected_float = img_corrected.astype(np.float32)
    img_reference_float = img_reference.astype(np.float32)
    
    # 转换到LAB色彩空间进行更精确的调整
    corrected_lab = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2LAB)
    reference_lab = cv2.cvtColor(img_reference, cv2.COLOR_BGR2LAB)
    
    # 提取重叠区域
    corrected_overlap = corrected_lab[valid_mask]
    reference_overlap = reference_lab[valid_mask]
    
    result_lab = corrected_lab.copy().astype(np.float32)
    
    # 对A和B通道进行细微调整（色彩平衡）
    for channel in [1, 2]:  # A和B通道
        corr_channel = corrected_overlap[:, channel]
        ref_channel = reference_overlap[:, channel]
        
        # 计算通道差异
        mean_diff = np.mean(ref_channel) - np.mean(corr_channel)
        
        # 应用轻微的色彩调整
        adjustment = mean_diff * 0.3  # 30%的调整强度
        result_lab[:, :, channel] += adjustment
    
    # 转换回BGR
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    
    return result_bgr


def create_adaptive_blend_weights(mask, img1, img2):
    """
    创建自适应的融合权重，考虑图像内容和几何因素
    """
    h, w = mask.shape
    weights = np.zeros((h, w), dtype=np.float32)
    valid_region = mask > 0
    
    if not np.any(valid_region):
        return weights
    
    # 计算到边界的距离变换
    distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # 找到重叠区域的边界
    y_coords, x_coords = np.where(valid_region)
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # 计算图像质量指标
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 计算局部方差 (作为纹理丰富度指标)
    kernel = np.ones((9, 9), np.float32) / 81
    var1 = cv2.filter2D(img1_gray.astype(np.float32), -1, kernel)
    var2 = cv2.filter2D(img2_gray.astype(np.float32), -1, kernel)
    
    # 为每个像素计算融合权重
    for y, x in zip(y_coords, x_coords):
        # 1. 基于位置的权重（从左到右渐变）
        if x_max > x_min:
            position_weight = (x - x_min) / (x_max - x_min)
        else:
            position_weight = 0.5
        
        # 2. 基于距离变换的稳定性权重
        distance_weight = min(distance_transform[y, x] / 20.0, 1.0)
        
        # 3. 基于图像质量的权重
        quality_weight = 0.5
        if var1[y, x] > var2[y, x]:
            quality_weight = 0.4  # 偏向纹理更丰富的img1
        elif var2[y, x] > var1[y, x]:
            quality_weight = 0.6  # 偏向纹理更丰富的img2
        
        # 结合所有权重
        combined_weight = (
            0.5 * position_weight +           # 50% 位置权重
            0.3 * quality_weight +            # 30% 质量权重  
            0.2 * distance_weight             # 20% 距离权重
        )
        
        # 使用平滑的S型曲线
        smooth_weight = 3 * combined_weight**2 - 2 * combined_weight**3
        weights[y, x] = np.clip(smooth_weight, 0.1, 0.9)
    
    # 高斯平滑使权重过渡更自然
    weights = cv2.GaussianBlur(weights, (21, 21), 7.0)
    
    return weights


def enhanced_multiband_pyramid_blend(img1, img2, weights, levels):
    """
    增强的多频带拉普拉斯金字塔融合
    """
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)
    
    # 构建拉普拉斯金字塔
    pyr1 = build_laplacian_pyramid_enhanced(img1_float, levels)
    pyr2 = build_laplacian_pyramid_enhanced(img2_float, levels)
    
    # 构建权重金字塔
    weight_pyr = build_gaussian_pyramid_enhanced(weights, levels)
    
    # 在每个层级上进行融合
    blended_pyr = []
    for level in range(levels):
        if level < len(pyr1) and level < len(pyr2) and level < len(weight_pyr):
            l1, l2 = pyr1[level], pyr2[level]
            w = weight_pyr[level]
            
            # 确保权重有正确的通道数
            if len(w.shape) == 2:
                w = np.stack([w] * 3, axis=2)
            
            # 频率自适应融合
            if level == 0:  # 最低频（粗糙层）
                # 低频使用平滑融合
                blended_level = l1 * (1 - w) + l2 * w
            else:  # 高频层
                # 高频层使用保留细节的融合
                blended_level = detail_preserving_blend(l1, l2, w, level)
            
            blended_pyr.append(blended_level)
        else:
            if pyr1:
                blended_pyr.append(pyr1[-1])
    
    # 重建图像
    result = reconstruct_from_laplacian_pyramid_enhanced(blended_pyr)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def detail_preserving_blend(l1, l2, weights, level):
    """
    保留细节的高频融合
    """
    # 计算局部能量来决定细节保留策略
    energy1 = np.sum(l1**2, axis=2)
    energy2 = np.sum(l2**2, axis=2)
    
    # 创建细节保留权重
    detail_weights = np.zeros_like(weights[:,:,0])
    detail_weights[energy1 > energy2] = 0.3  # 偏向细节更丰富的区域
    detail_weights[energy2 > energy1] = 0.7
    detail_weights[np.abs(energy1 - energy2) < 0.1 * np.maximum(energy1, energy2)] = 0.5
    
    # 结合原始权重和细节权重
    combined_weights = 0.6 * weights[:,:,0] + 0.4 * detail_weights
    combined_weights = np.stack([combined_weights] * 3, axis=2)
    
    return l1 * (1 - combined_weights) + l2 * combined_weights


def build_laplacian_pyramid_enhanced(img, levels):
    """
    构建增强的拉普拉斯金字塔，使用更好的核函数
    """
    gaussian_pyr = build_gaussian_pyramid_enhanced(img, levels)
    laplacian_pyr = []
    
    for i in range(levels - 1):
        # 上采样下一层
        expanded = cv2.pyrUp(gaussian_pyr[i + 1])
        
        # 确保尺寸匹配
        if expanded.shape[:2] != gaussian_pyr[i].shape[:2]:
            expanded = cv2.resize(expanded, 
                                (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0]))
        
        # 计算拉普拉斯层
        laplacian = gaussian_pyr[i] - expanded
        laplacian_pyr.append(laplacian)
    
    # 添加最后一层
    laplacian_pyr.append(gaussian_pyr[-1])
    
    return laplacian_pyr


def build_gaussian_pyramid_enhanced(img, levels):
    """
    构建增强的高斯金字塔，使用改进的滤波器
    """
    pyramid = [img.copy()]
    current = img.copy()
    
    for i in range(levels - 1):
        # 使用5x5高斯核进行更好的抗锯齿
        current = cv2.GaussianBlur(current, (5, 5), 1.0)
        current = cv2.pyrDown(current)
        pyramid.append(current)
    
    return pyramid


def reconstruct_from_laplacian_pyramid_enhanced(laplacian_pyr):
    """
    从增强的拉普拉斯金字塔重建图像
    """
    current = laplacian_pyr[-1].copy()
    
    for i in range(len(laplacian_pyr) - 2, -1, -1):
        # 上采样
        current = cv2.pyrUp(current)
        
        # 确保尺寸匹配
        if current.shape[:2] != laplacian_pyr[i].shape[:2]:
            current = cv2.resize(current, 
                               (laplacian_pyr[i].shape[1], laplacian_pyr[i].shape[0]))
        
        # 添加拉普拉斯层
        current = current + laplacian_pyr[i]
    
    return current


def post_process_blend(img, img1, img2, mask):
    """
    后处理优化，进一步减少接缝和色彩不连续性
    """
    result = img.copy().astype(np.float32)
    
    # 在边界附近应用额外的平滑
    boundary_mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((7,7), np.uint8))
    boundary_mask = boundary_mask > 0
    
    if np.any(boundary_mask):
        # 对边界区域应用轻微的边缘保持平滑
        for channel in range(3):
            boundary_region = result[:,:,channel]
            # 使用双边滤波保持边缘的同时平滑色彩过渡
            smoothed = cv2.bilateralFilter(boundary_region.astype(np.uint8), 7, 40, 40)
            # 轻微混合
            result[boundary_mask, channel] = (
                result[boundary_mask, channel] * 0.8 + 
                smoothed[boundary_mask] * 0.2
            )
    
    return np.clip(result, 0, 255).astype(np.uint8)


def gain_compensation(img_source, img_target, overlap_mask=None):
    """
    增强版增益补偿 (Gain Compensation) 算法
    基于 Brown & Lowe 2007 论文，添加了更强的色彩校正
    这是解决色差问题的foundational step
    """
    if overlap_mask is None:
        # 如果没有提供重叠区域mask，使用整个图像
        h, w = img_source.shape[:2]
        overlap_mask = np.ones((h, w), dtype=np.uint8) * 255
    
    valid_mask = overlap_mask > 0
    if not np.any(valid_mask):
        return img_source
    
    # 第一步：在LAB色彩空间进行初步校正
    img_source_lab = cv2.cvtColor(img_source, cv2.COLOR_BGR2LAB).astype(np.float64)
    img_target_lab = cv2.cvtColor(img_target, cv2.COLOR_BGR2LAB).astype(np.float64)
    
    # 提取重叠区域
    source_overlap = img_source_lab[valid_mask]
    target_overlap = img_target_lab[valid_mask]
    
    result_lab = img_source_lab.copy()
    
    # 对LAB三个通道分别处理
    for channel in range(3):
        source_channel = source_overlap[:, channel]
        target_channel = target_overlap[:, channel]
        
        # 更严格的有效像素筛选
        if channel == 0:  # L通道 (亮度)
            valid_pixels = (source_channel > 10) & (source_channel < 90) & \
                          (target_channel > 10) & (target_channel < 90)
        else:  # A和B通道 (色彩)
            valid_pixels = (source_channel > 110) & (source_channel < 150) & \
                          (target_channel > 110) & (target_channel < 150)
        
        if np.sum(valid_pixels) < 50:
            # 如果有效像素太少，使用所有像素
            valid_pixels = np.ones(len(source_channel), dtype=bool)
        
        source_valid = source_channel[valid_pixels]
        target_valid = target_channel[valid_pixels]
        
        if len(source_valid) == 0:
            continue
        
        # 使用多种统计方法的加权组合
        # 方法1: 均值匹配
        source_mean = np.mean(source_valid)
        target_mean = np.mean(target_valid)
        mean_diff = target_mean - source_mean
        
        # 方法2: 中位数匹配 (更鲁棒)
        source_median = np.median(source_valid)
        target_median = np.median(target_valid)
        median_diff = target_median - source_median
        
        # 方法3: 分位数匹配
        source_q25 = np.percentile(source_valid, 25)
        source_q75 = np.percentile(source_valid, 75)
        target_q25 = np.percentile(target_valid, 25)
        target_q75 = np.percentile(target_valid, 75)
        
        # 计算增益和偏移
        source_range = source_q75 - source_q25
        target_range = target_q75 - target_q25
        
        if source_range > 1e-6:
            gain = target_range / source_range
            gain = np.clip(gain, 0.5, 2.0)  # 更保守的增益范围
        else:
            gain = 1.0
        
        # 综合多种方法计算偏移
        offset_mean = mean_diff
        offset_median = median_diff  
        offset_percentile = target_q25 - gain * source_q25
        
        # 加权平均，中位数权重更高（更鲁棒）
        offset = 0.2 * offset_mean + 0.6 * offset_median + 0.2 * offset_percentile
        
        # 限制偏移范围
        if channel == 0:  # L通道
            offset = np.clip(offset, -30, 30)
        else:  # A和B通道
            offset = np.clip(offset, -20, 20)
        
        # 应用变换
        result_lab[:, :, channel] = gain * img_source_lab[:, :, channel] + offset
    
    # 确保LAB值在有效范围内
    result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 100)  # L通道
    result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], -127, 127)  # A通道
    result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], -127, 127)  # B通道
    
    # 转换回BGR
    result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # 第二步：在BGR空间进行微调
    result_bgr_float = result_bgr.astype(np.float64)
    img_target_float = img_target.astype(np.float64)
    
    # 计算整体色温调整
    result_mean = np.mean(result_bgr_float[valid_mask], axis=0)
    target_mean = np.mean(img_target_float[valid_mask], axis=0)
    
    # 轻微调整以进一步匹配
    color_diff = target_mean - result_mean
    adjustment = color_diff * 0.3  # 30%的调整强度
    
    result_bgr_float += adjustment
    
    return np.clip(result_bgr_float, 0, 255).astype(np.uint8)


def exposure_compensation(img2, img1, mask):
    """
    曝光补偿算法，基于论文中的线性变换模型
    已被gain_compensation取代，保留用于兼容性
    """
    return gain_compensation(img2, img1, mask)


def create_optimal_blend_mask(binary_mask, img1, img2):
    """
    创建优化的融合mask，考虑图像内容和几何因素
    """
    h, w = binary_mask.shape
    mask = np.zeros((h, w), dtype=np.float32)
    valid_region = binary_mask > 0
    
    if not np.any(valid_region):
        return mask
    
    # 计算到边界的距离
    distance_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    
    # 找到重叠区域的中心线
    y_coords, x_coords = np.where(valid_region)
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    
    # 计算每个像素的权重
    for y, x in zip(y_coords, x_coords):
        # 基于位置的权重（从左到右渐变）
        if x_max > x_min:
            position_weight = (x - x_min) / (x_max - x_min)
        else:
            position_weight = 0.5
        
        # 基于距离变换的权重（远离边界的区域权重更稳定）
        distance_weight = min(distance_transform[y, x] / 20.0, 1.0)
        
        # 结合权重，使用平滑的S型曲线
        combined_weight = position_weight
        combined_weight = 3 * combined_weight**2 - 2 * combined_weight**3  # S型曲线
        
        # 应用距离权重的影响
        final_weight = combined_weight * (0.5 + 0.5 * distance_weight)
        mask[y, x] = np.clip(final_weight, 0.1, 0.9)
    
    # 高斯平滑，使权重过渡更自然
    mask = cv2.GaussianBlur(mask, (15, 15), 5.0)
    
    return mask


def multiband_pyramid_blend(img1, img2, mask, levels):
    """
    多频带拉普拉斯金字塔融合
    基于经典论文方法，但使用现代优化
    """
    # 确保图像尺寸一致
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # 构建拉普拉斯金字塔
    pyr1 = build_laplacian_pyramid_advanced(img1, levels)
    pyr2 = build_laplacian_pyramid_advanced(img2, levels)
    
    # 构建mask金字塔
    mask_pyr = build_gaussian_pyramid_advanced(mask, levels)
    
    # 在每个层级上进行融合
    blended_pyr = []
    for level in range(levels):
        if level < len(pyr1) and level < len(pyr2) and level < len(mask_pyr):
            # 获取当前层级的图像和mask
            l1, l2 = pyr1[level], pyr2[level]
            m = mask_pyr[level]
            
            # 确保mask有正确的通道数
            if len(m.shape) == 2:
                m = np.stack([m] * 3, axis=2)
            
            # 自适应融合权重，在不同频率上使用不同策略
            if level == 0:  # 低频（最粗糙层）
                # 低频使用平滑融合
                blended_level = l1 * (1 - m) + l2 * m
            else:  # 高频层
                # 高频层使用更智能的融合
                blended_level = adaptive_frequency_blend(l1, l2, m, level)
            
            blended_pyr.append(blended_level)
        else:
            # 如果层级不够，使用可用的最后一层
            if pyr1:
                blended_pyr.append(pyr1[-1])
    
    # 重建图像
    result = reconstruct_from_laplacian_pyramid_advanced(blended_pyr)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def adaptive_frequency_blend(l1, l2, mask, level):
    """
    自适应频率融合，在不同频率上使用不同策略
    """
    # 计算局部方差来确定哪个图像在当前区域有更好的细节
    kernel_size = max(3, 2*level + 1)
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    
    # 计算局部方差
    var1 = cv2.filter2D((l1**2).mean(axis=2), -1, kernel) - cv2.filter2D(l1.mean(axis=2), -1, kernel)**2
    var2 = cv2.filter2D((l2**2).mean(axis=2), -1, kernel) - cv2.filter2D(l2.mean(axis=2), -1, kernel)**2
    
    # 创建基于细节保留的权重
    detail_weight = np.zeros_like(mask[:,:,0])
    detail_weight[var1 > var2] = 0.3  # 偏向细节更丰富的图像
    detail_weight[var2 > var1] = 0.7
    detail_weight[np.abs(var1 - var2) < np.mean([var1, var2]) * 0.1] = 0.5  # 相似区域使用平均权重
    
    # 结合原始mask和细节权重
    combined_mask = 0.7 * mask[:,:,0] + 0.3 * detail_weight
    combined_mask = np.stack([combined_mask] * 3, axis=2)
    
    return l1 * (1 - combined_mask) + l2 * combined_mask


def gradient_domain_optimization(img, img1, img2, mask):
    """
    梯度域优化，减少接缝和色彩不连续性
    基于泊松编辑的思想
    """
    # 简化版的梯度域优化
    result = img.copy().astype(np.float32)
    valid_mask = mask > 0
    
    if not np.any(valid_mask):
        return result.astype(np.uint8)
    
    # 在边界附近应用额外的平滑
    boundary_mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((5,5), np.uint8))
    boundary_mask = boundary_mask > 0
    
    if np.any(boundary_mask):
        # 对边界区域应用轻微的双边滤波
        for channel in range(3):
            boundary_region = result[:,:,channel]
            smoothed = cv2.bilateralFilter(boundary_region.astype(np.uint8), 5, 30, 30)
            result[boundary_mask, channel] = (
                result[boundary_mask, channel] * 0.7 + 
                smoothed[boundary_mask] * 0.3
            )
    
    return np.clip(result, 0, 255).astype(np.uint8)


def build_laplacian_pyramid_advanced(img, levels):
    """
    构建改进的拉普拉斯金字塔
    """
    gaussian_pyr = build_gaussian_pyramid_advanced(img, levels)
    laplacian_pyr = []
    
    for i in range(levels - 1):
        # 上采样下一层
        expanded = cv2.pyrUp(gaussian_pyr[i + 1])
        
        # 确保尺寸匹配
        if expanded.shape[:2] != gaussian_pyr[i].shape[:2]:
            expanded = cv2.resize(expanded, (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0]))
        
        # 计算拉普拉斯层
        laplacian = gaussian_pyr[i] - expanded
        laplacian_pyr.append(laplacian)
    
    # 添加最后一层（最小的高斯层）
    laplacian_pyr.append(gaussian_pyr[-1])
    
    return laplacian_pyr


def build_gaussian_pyramid_advanced(img, levels):
    """
    构建改进的高斯金字塔，使用更好的下采样核
    """
    pyramid = [img.copy()]
    current = img.copy()
    
    for i in range(levels - 1):
        # 使用更好的抗锯齿核
        current = cv2.GaussianBlur(current, (5, 5), 0.8)
        current = cv2.pyrDown(current)
        pyramid.append(current)
    
    return pyramid


def reconstruct_from_laplacian_pyramid_advanced(laplacian_pyr):
    """
    从改进的拉普拉斯金字塔重建图像
    """
    current = laplacian_pyr[-1].copy()
    
    for i in range(len(laplacian_pyr) - 2, -1, -1):
        # 上采样
        current = cv2.pyrUp(current)
        
        # 确保尺寸匹配
        if current.shape[:2] != laplacian_pyr[i].shape[:2]:
            current = cv2.resize(current, (laplacian_pyr[i].shape[1], laplacian_pyr[i].shape[0]))
        
        # 添加拉普拉斯层
        current = current + laplacian_pyr[i]
    
    return current


def simple_effective_blend(img1, img2):
    """
    简单有效的融合算法，重点解决色差问题
    """
    h, w = img1.shape[:2]
    
    # 1. 白平衡校正
    img2_wb = white_balance_correction(img2, img1)
    
    # 2. 拉普拉斯金字塔融合
    blended = laplacian_pyramid_blend(img1, img2_wb)
    
    return blended


def white_balance_correction(source, reference):
    """
    简单的白平衡校正
    """
    source_float = source.astype(np.float32)
    reference_float = reference.astype(np.float32)
    
    # 计算每个通道的平均值
    source_mean = np.mean(source_float, axis=(0, 1))
    reference_mean = np.mean(reference_float, axis=(0, 1))
    
    # 计算校正比例，避免除零
    ratios = np.where(source_mean > 1, reference_mean / source_mean, 1.0)
    
    # 应用校正
    corrected = source_float * ratios
    
    return np.clip(corrected, 0, 255).astype(np.uint8)


def laplacian_pyramid_blend(img1, img2, levels=4):
    """
    拉普拉斯金字塔融合，产生无缝结果
    """
    h, w = img1.shape[:2]
    
    # 创建混合mask（中心权重更高）
    mask = np.ones((h, w), dtype=np.float32)
    center_w = w // 2
    
    # 从左到右的渐变权重
    for x in range(w):
        if x < center_w:
            weight = 0.3 + 0.4 * (x / center_w)  # 从0.3渐变到0.7
        else:
            weight = 0.7 + 0.3 * ((w - 1 - x) / (w - center_w))  # 从0.7渐变到1.0
        mask[:, x] = weight
    
    # 构建高斯金字塔
    pyr1 = build_gaussian_pyramid(img1.astype(np.float32), levels)
    pyr2 = build_gaussian_pyramid(img2.astype(np.float32), levels)
    mask_pyr = build_gaussian_pyramid(mask, levels)
    
    # 构建拉普拉斯金字塔
    lap1 = build_laplacian_pyramid(pyr1)
    lap2 = build_laplacian_pyramid(pyr2)
    
    # 混合拉普拉斯金字塔
    blended_lap = []
    for i in range(levels):
        if i < len(mask_pyr):
            m = mask_pyr[i]
            if len(m.shape) == 2:
                m = np.stack([m] * 3, axis=2)
            blended_lap.append(lap1[i] * m + lap2[i] * (1 - m))
        else:
            blended_lap.append(lap1[i] * 0.5 + lap2[i] * 0.5)
    
    # 重建图像
    result = reconstruct_from_laplacian_pyramid(blended_lap)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def build_gaussian_pyramid(img, levels):
    """构建高斯金字塔"""
    pyramid = [img]
    current = img.copy()
    
    for i in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    
    return pyramid


def build_laplacian_pyramid(gaussian_pyramid):
    """构建拉普拉斯金字塔"""
    laplacian_pyramid = []
    
    for i in range(len(gaussian_pyramid) - 1):
        # 上采样下一层
        expanded = cv2.pyrUp(gaussian_pyramid[i + 1])
        
        # 确保尺寸匹配
        if expanded.shape[:2] != gaussian_pyramid[i].shape[:2]:
            expanded = cv2.resize(expanded, 
                                (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        
        # 计算拉普拉斯层
        laplacian = gaussian_pyramid[i] - expanded
        laplacian_pyramid.append(laplacian)
    
    # 最后一层直接添加
    laplacian_pyramid.append(gaussian_pyramid[-1])
    
    return laplacian_pyramid


def reconstruct_from_laplacian_pyramid(laplacian_pyramid):
    """从拉普拉斯金字塔重建图像"""
    current = laplacian_pyramid[-1]
    
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        # 上采样
        current = cv2.pyrUp(current)
        
        # 确保尺寸匹配
        if current.shape[:2] != laplacian_pyramid[i].shape[:2]:
            current = cv2.resize(current, 
                               (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]))
        
        # 添加拉普拉斯层
        current = current + laplacian_pyramid[i]
    
    return current


def color_match(source, target):
    """
    使用直方图匹配进行色彩校正
    """
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    
    matched_lab = np.zeros_like(source_lab)
    
    # 对每个LAB通道进行直方图匹配
    for i in range(3):
        source_hist = cv2.calcHist([source_lab], [i], None, [256], [0, 256])
        target_hist = cv2.calcHist([target_lab], [i], None, [256], [0, 256])
        
        # 计算累积分布函数
        source_cdf = source_hist.cumsum()
        target_cdf = target_hist.cumsum()
        
        # 标准化
        source_cdf = source_cdf / source_cdf[-1]
        target_cdf = target_cdf / target_cdf[-1]
        
        # 创建查找表
        lut = np.zeros(256, dtype=np.uint8)
        for j in range(256):
            idx = np.argmin(np.abs(target_cdf - source_cdf[j]))
            lut[j] = idx
        
        matched_lab[:, :, i] = cv2.LUT(source_lab[:, :, i], lut)
    
    return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)


def gradient_blend(img1, img2, blend_width=50):
    """
    创建渐变融合效果减少接缝处的色差
    """
    h, w = img1.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)
    
    # 创建从左到右的渐变权重
    if w > blend_width * 2:
        # 左边缘渐变
        for i in range(min(blend_width, w//2)):
            mask[:, i] = i / blend_width
        # 右边缘渐变  
        for i in range(max(0, w - blend_width), w):
            mask[:, i] = (w - 1 - i) / blend_width
    
    # 扩展mask到3个通道
    mask = np.stack([mask] * 3, axis=2)
    
    # 进行加权混合
    blended = img1.astype(np.float32) * mask + img2.astype(np.float32) * (1 - mask)
    
    return np.uint8(np.clip(blended, 0, 255))


@jit(nopython=True)
def combine_overlap1(segments, img1, img2, overlap, seam, seam_output):
    # 行，列

    for i in range(segments.shape[0]):
        for j in range(segments.shape[1]):
            if not segments[i][j]:
                overlap[i][j] = img2[i][j]
            else:
                overlap[i][j] = img1[i][j]
    for i in range(seam.shape[0]):
        for j in range(seam.shape[1]):
            if not seam[i][j]:
                seam_output[i][j] = overlap[i][j]
            else:
                seam_output[i][j] = [0, 255, 0]

    return overlap, seam_output
