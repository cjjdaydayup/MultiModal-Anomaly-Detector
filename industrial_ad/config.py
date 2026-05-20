from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_DEFECT_WORDS = [
    "scratch",
    "crack",
    "stain",
    "defect",
    "hole",
    "dent",
    "corrosion",
]


@dataclass
class DetectorConfig:
    """Runtime options for model loading and post-processing."""

    model_path: str = "models/clipseg"
    threshold: float = 0.72
    mask_threshold: int = 180
    min_region_area: int = 50
    device: str = "auto"
    defect_words: List[str] = field(default_factory=lambda: list(DEFAULT_DEFECT_WORDS))
    prompt_template: str = "{defect} on {product}"
    output_dir: str = "outputs"
    save_visuals: bool = True
    heatmap_alpha: float = 0.5

    def validate(self) -> None:
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if not 0 <= self.mask_threshold <= 255:
            raise ValueError("mask_threshold must be between 0 and 255")
        if self.min_region_area < 0:
            raise ValueError("min_region_area must not be negative")
        if not self.defect_words:
            raise ValueError("at least one defect word is required")
        if "{product}" not in self.prompt_template or "{defect}" not in self.prompt_template:
            raise ValueError("prompt_template must contain {product} and {defect}")
        if not 0 <= self.heatmap_alpha <= 1:
            raise ValueError("heatmap_alpha must be between 0 and 1")

    def build_prompts(self, product_name: str, extra_words: Optional[Iterable[str]] = None) -> List[str]:
        words = list(self.defect_words)
        if extra_words:
            for word in extra_words:
                word = word.strip()
                if word and word not in words:
                    words.append(word)
        return [
            self.prompt_template.format(defect=word, product=product_name)
            for word in words
        ]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectorConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        clean = {key: value for key, value in data.items() if key in allowed}
        config = cls(**clean)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "DetectorConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)


@dataclass
class BatchConfig:
    """Options used by batch detection jobs."""

    recursive: bool = True
    continue_on_error: bool = True
    extensions: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
    report_formats: List[str] = field(default_factory=lambda: ["json", "csv", "html"])

    def normalized_extensions(self) -> List[str]:
        result = []
        for value in self.extensions:
            extension = value.lower().strip()
            if not extension:
                continue
            if not extension.startswith("."):
                extension = "." + extension
            if extension not in result:
                result.append(extension)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: Optional[str | Path] = None) -> DetectorConfig:
    if path is None:
        return DetectorConfig()
    return DetectorConfig.load(path)
