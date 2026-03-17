import torch
import cv2
import numpy as np


def detect_universal_anomaly(image_pil, product_name, processor, model):
    """大模型零样本缺陷检测核心逻辑"""
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape

    prompts = [
        f"scratch on {product_name}",
        f"crack on {product_name}",
        f"stain on {product_name}",
        f"defect on {product_name}",
        f"hole on {product_name}"
    ]

    inputs = processor(
        text=prompts, images=[image_pil] * len(prompts),
        padding="max_length", return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    preds = outputs.logits.unsqueeze(1)
    combined_mask = torch.max(preds, dim=0)[0].squeeze()

    mask_np = torch.sigmoid(combined_mask).cpu().numpy()
    mask_normalized = cv2.normalize(mask_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_resized = cv2.resize(mask_normalized, (w, h))

    anomaly_score = float(np.max(mask_np))

    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(img_cv, 0.5, heatmap_color, 0.5, 0)

    boxed_img = img_cv.copy()
    _, thresh = cv2.threshold(heatmap_resized, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    is_defective = False
    if len(contours) > 0:
        for c in contours:
            if cv2.contourArea(c) > 50:
                is_defective = True
                x, y, bw, bh = cv2.boundingRect(c)
                cv2.rectangle(boxed_img, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
                cv2.putText(boxed_img, "Defect", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    overlay_img_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
    boxed_img_rgb = cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB)

    return is_defective, anomaly_score, overlay_img_rgb, boxed_img_rgb