import cv2
import numpy as np
import operator


def apply_contrast(image, level, alpha_min=0.01, alpha_max=10.0):
    """
    对比度扰动
    image   uint8 BGR
    level   对比度控制参数 取值在负一到正一
            level 等于零时不变
            正值拉高对比度
            负值压低对比度
    alpha_min  最小对比度倍率 例如三成对比度
    alpha_max  最大对比度倍率 例如四倍对比度
    """
    if level == 0.0:
        return image

    img = image.astype(np.float32) / 255.0

    # 全局均值
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    diff = np.subtract(img, mean)

    base = 1.0

    if level > 0.0:
        # 正方向 从一渐变到 alpha_max
        up_scale = operator.sub(alpha_max, base)
        alpha = base + float(level) * up_scale
    else:
        # 负方向 从一渐变到 alpha_min 而不是零
        down_scale = operator.sub(base, alpha_min)
        alpha = base + float(level) * down_scale

    # 多一层保险 防止数值跑出范围
    alpha = float(np.clip(alpha, alpha_min, alpha_max))

    img = diff * alpha + mean
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)

def apply_blur(image, b_blur, sigma_max=10.0):
    """
    高斯模糊
    b_blur   归一化模糊强度 0 到 1
             0 为无模糊 1 对应最大模糊 sigma_max
    """
    if b_blur <= 0.0:
        return image

    img = image.astype(np.float32) / 255.0

    sigma = float(b_blur) * float(sigma_max)

    ksize = int(sigma * 6.0)
    if ksize < 3:
        ksize = 3
    if ksize % 2 == 0:
        ksize = ksize + 1

    img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def apply_fog(image, b_fog, fog_max=0.99):
    """
    雾效扰动
    b_fog   归一化雾强度 0 到 1
            0 为无雾 1 对应最大雾强度 fog_max
    """
    if b_fog <= 0.0:
        return image

    img = image.astype(np.float32) / 255.0

    strength = float(b_fog) * float(fog_max)
    strength = float(np.clip(strength, 0.0, 1.0))

    fog_layer = np.ones_like(img)
    img = img * (1.0 - strength) + fog_layer * strength

    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def apply_brightness(image, b_brightness, factor_min=0.01, factor_max=20.0):
    """
    亮度扰动
    image        uint8 BGR
    b_brightness 归一化亮度参数 取值在负一到正一
                 零为不变 大于零变亮 小于零变暗
    factor_min   最暗时的亮度倍数 例如零点一
    factor_max   最亮时的亮度倍数 例如六
    """
    img = image.astype(np.float32) / 255.0

    base = 1.0

    if b_brightness >= 0.0:
        # 正方向: 从一插值到 factor_max
        up_span = np.subtract(factor_max, base)
        factor = base + float(b_brightness) * up_span
    else:
        # 负方向: 从一插值到 factor_min
        down_span = np.subtract(base, factor_min)
        factor = base + float(b_brightness) * down_span

    factor = float(np.clip(factor, factor_min, factor_max))

    img = img * factor
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)

# img = cv2.imread("test.png")
# img = img.astype(np.uint8)

# img = apply_brightness(img, 0)

# # 对比度
# img = apply_contrast(img, 0)

# # 雾
# img = apply_fog(img, 0)

# # 模糊
# img = apply_blur(img, 1)
# cv2.imwrite("output_blurred.png", img)
