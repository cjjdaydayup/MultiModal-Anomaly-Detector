import cv2
import numpy as np
from PIL import Image


def adjust_brightness_contrast(image_pil, brightness=0, contrast=0):
    """
    调整图像亮度与对比度
    :param brightness: 亮度 (-127 到 127)
    :param contrast: 对比度 (-127 到 127)
    """
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv2.addWeighted(img_cv, alpha_b, img_cv, 0, gamma_b)
    else:
        buf = img_cv.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return Image.fromarray(cv2.cvtColor(buf, cv2.COLOR_BGR2RGB))


def apply_clahe(image_pil, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE (限制对比度自适应直方图均衡化) - 工业界极其常用的光照不均解决方案
    """
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    # 转换到 LAB 颜色空间，只处理 L (亮度) 通道
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))


def apply_denoise(image_pil, strength=10):
    """
    非局部均值去噪 - 用于去除工业相机由于高ISO产生的噪点
    """
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    dst = cv2.fastNlMeansDenoisingColored(img_cv, None, strength, strength, 7, 21)
    return Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))