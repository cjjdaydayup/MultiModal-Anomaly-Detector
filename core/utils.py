# core/utils.py
import os
import json
import logging
from datetime import datetime
import cv2
import numpy as np
import io
from PIL import Image


def setup_logger(log_dir="logs"):
    """
    配置系统独立日志模块，记录每一次检测操作，用于工业审计溯源
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"audit_log_{datetime.now().strftime('%Y%m%d')}.log")

    logger = logging.getLogger("IndustrialAD")
    logger.setLevel(logging.INFO)

    # 防止重复添加 handler
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def save_detection_record(product_name, is_defective, score, save_dir="records"):
    """
    持久化存储检测记录到本地 JSON 数据库中
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    record_file = os.path.join(save_dir, "detection_history.json")

    new_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "product": product_name,
        "result": "Defective" if is_defective else "Normal",
        "confidence_score": round(score, 4)
    }

    records = []
    if os.path.exists(record_file):
        try:
            with open(record_file, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            pass

    records.append(new_record)

    with open(record_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)

    return True


# 预留给后期批量检测的辅助函数
def resize_and_pad(image_cv, target_size=(512, 512)):
    """保持长宽比的图像缩放与填充"""
    h, w = image_cv.shape[:2]
    sh, sw = target_size
    aspect = w / h

    if aspect > 1:
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
    else:
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)

    resized = cv2.resize(image_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_h = sh - new_h
    pad_w = sw - new_w
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded