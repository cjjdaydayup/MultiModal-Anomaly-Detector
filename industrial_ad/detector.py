from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from industrial_ad.config import DetectorConfig
from industrial_ad.types import DetectionRegion, DetectionResult


class ModelLoadError(RuntimeError):
    """Raised when the local model cannot be loaded."""


class AnomalyDetector:
    """Importable detector API for third-party Python programs.

    Example:
        detector = AnomalyDetector("models/clipseg")
        result = detector.detect_path("sample_images/bottle1.png", "bottle")
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        config: Optional[DetectorConfig] = None,
        processor: Any = None,
        model: Any = None,
    ) -> None:
        self.config = config or DetectorConfig()
        if model_path is not None:
            self.config.model_path = str(model_path)
        self.config.validate()
        self.processor = processor
        self.model = model
        self._loaded = bool(processor is not None and model is not None)

    def load(self) -> "AnomalyDetector":
        if self._loaded:
            return self
        try:
            from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        except Exception as exc:
            raise ModelLoadError(
                "transformers is required for AnomalyDetector.load(); install project requirements first"
            ) from exc

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise ModelLoadError(
                f"model path does not exist: {model_path}. Run download_model.py or pass a valid model path."
            )

        self.processor = CLIPSegProcessor.from_pretrained(str(model_path))
        self.model = CLIPSegForImageSegmentation.from_pretrained(str(model_path))
        self._loaded = True
        return self

    def detect_path(
        self,
        image_path: str | Path,
        product_name: str,
        extra_defect_words: Optional[Iterable[str]] = None,
        output_dir: Optional[str | Path] = None,
    ) -> DetectionResult:
        path = Path(image_path)
        image = Image.open(path).convert("RGB")
        result = self.detect_image(image, product_name, extra_defect_words, output_dir, str(path))
        return result

    def detect_array(
        self,
        image_array: np.ndarray,
        product_name: str,
        extra_defect_words: Optional[Iterable[str]] = None,
        output_dir: Optional[str | Path] = None,
        source_path: Optional[str] = None,
    ) -> DetectionResult:
        if image_array.ndim != 3:
            raise ValueError("image_array must be an RGB or BGR image with 3 channels")
        image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        return self.detect_image(image, product_name, extra_defect_words, output_dir, source_path)

    def detect_image(
        self,
        image: Image.Image,
        product_name: str,
        extra_defect_words: Optional[Iterable[str]] = None,
        output_dir: Optional[str | Path] = None,
        source_path: Optional[str] = None,
    ) -> DetectionResult:
        product_name = product_name.strip()
        if not product_name:
            raise ValueError("product_name must not be empty")
        if image.mode != "RGB":
            image = image.convert("RGB")

        self.load()
        prompts = self.config.build_prompts(product_name, extra_defect_words)
        mask = self._predict_mask(image, prompts)
        return self._build_result(image, mask, product_name, prompts, source_path, output_dir)

    def _predict_mask(self, image: Image.Image, prompts: list[str]) -> np.ndarray:
        import torch

        inputs = self.processor(
            text=prompts,
            images=[image] * len(prompts),
            padding="max_length",
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        preds = outputs.logits.unsqueeze(1)
        combined_mask = torch.max(preds, dim=0)[0].squeeze()
        return torch.sigmoid(combined_mask).cpu().numpy()

    def _build_result(
        self,
        image: Image.Image,
        mask: np.ndarray,
        product_name: str,
        prompts: list[str],
        source_path: Optional[str],
        output_dir: Optional[str | Path],
    ) -> DetectionResult:
        rgb = np.array(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        heatmap_gray = self._normalize_mask(mask, (width, height))
        score = float(np.max(mask))
        regions = self._extract_regions(heatmap_gray, mask)
        is_defective = score >= self.config.threshold and bool(regions)

        result = DetectionResult(
            product_name=product_name,
            is_defective=is_defective,
            score=score,
            threshold=self.config.threshold,
            image_size=(width, height),
            prompts=prompts,
            regions=regions,
            source_path=source_path,
            metadata={
                "mask_threshold": self.config.mask_threshold,
                "min_region_area": self.config.min_region_area,
            },
        )

        if self.config.save_visuals:
            target_dir = Path(output_dir or self.config.output_dir)
            heatmap, boxed = self.render_visuals(image, heatmap_gray, regions)
            heatmap_path, boxed_path = self._save_visuals(target_dir, source_path, heatmap, boxed)
            result.with_outputs(str(heatmap_path), str(boxed_path))

        return result

    def _normalize_mask(self, mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        normalized = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.resize(normalized, size)

    def _extract_regions(self, heatmap_gray: np.ndarray, mask: np.ndarray) -> list[DetectionRegion]:
        _, thresh = cv2.threshold(
            heatmap_gray,
            self.config.mask_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        global_score = float(np.max(mask))
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.config.min_region_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            regions.append(
                DetectionRegion(
                    box=(int(x), int(y), int(width), int(height)),
                    area=area,
                    score=global_score,
                )
            )
        regions.sort(key=lambda item: item.area, reverse=True)
        return regions

    def render_visuals(
        self,
        image: Image.Image,
        heatmap_gray: np.ndarray,
        regions: list[DetectionRegion],
    ) -> Tuple[np.ndarray, np.ndarray]:
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        alpha = self.config.heatmap_alpha
        overlay = cv2.addWeighted(bgr, 1 - alpha, heatmap_color, alpha, 0)

        boxed = bgr.copy()
        for index, region in enumerate(regions, start=1):
            x, y, width, height = region.box
            cv2.rectangle(boxed, (x, y), (x + width, y + height), (0, 0, 255), 3)
            cv2.putText(
                boxed,
                f"Defect {index}",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)

    def _save_visuals(
        self,
        output_dir: Path,
        source_path: Optional[str],
        heatmap: np.ndarray,
        boxed: np.ndarray,
    ) -> Tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = self._make_output_stem(source_path)
        heatmap_path = output_dir / f"{stem}_heatmap.jpg"
        boxed_path = output_dir / f"{stem}_boxed.jpg"
        Image.fromarray(heatmap).save(heatmap_path, format="JPEG", quality=92)
        Image.fromarray(boxed).save(boxed_path, format="JPEG", quality=92)
        return heatmap_path, boxed_path

    def _make_output_stem(self, source_path: Optional[str]) -> str:
        if source_path:
            base = Path(source_path).stem
            digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:8]
            return f"{base}_{digest}"
        return hashlib.sha1(str(id(self)).encode("utf-8")).hexdigest()[:12]

    def explain_config(self) -> Dict[str, Any]:
        return self.config.to_dict()
