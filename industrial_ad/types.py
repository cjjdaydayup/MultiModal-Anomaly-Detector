from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


Box = Tuple[int, int, int, int]


def _iso_now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


@dataclass(frozen=True)
class DetectionRegion:
    """A single defect-like area found in an image."""

    box: Box
    area: float
    score: float
    label: str = "defect"

    @property
    def x(self) -> int:
        return self.box[0]

    @property
    def y(self) -> int:
        return self.box[1]

    @property
    def width(self) -> int:
        return self.box[2]

    @property
    def height(self) -> int:
        return self.box[3]

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "box": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
            "area": _round_float(self.area, 3),
            "score": _round_float(self.score),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionRegion":
        raw_box = data.get("box", {})
        if isinstance(raw_box, dict):
            box = (
                int(raw_box.get("x", 0)),
                int(raw_box.get("y", 0)),
                int(raw_box.get("width", 0)),
                int(raw_box.get("height", 0)),
            )
        else:
            box = tuple(raw_box)  # type: ignore[assignment]
        return cls(
            box=box,
            area=float(data.get("area", 0)),
            score=float(data.get("score", 0)),
            label=str(data.get("label", "defect")),
        )


@dataclass
class DetectionResult:
    """Structured result returned by AnomalyDetector."""

    product_name: str
    is_defective: bool
    score: float
    threshold: float
    image_size: Tuple[int, int]
    prompts: List[str] = field(default_factory=list)
    regions: List[DetectionRegion] = field(default_factory=list)
    source_path: Optional[str] = None
    heatmap_path: Optional[str] = None
    boxed_path: Optional[str] = None
    created_at: str = field(default_factory=_iso_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "Defective" if self.is_defective else "Normal"

    @property
    def region_count(self) -> int:
        return len(self.regions)

    def with_outputs(self, heatmap_path: Optional[str], boxed_path: Optional[str]) -> "DetectionResult":
        self.heatmap_path = heatmap_path
        self.boxed_path = boxed_path
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_name": self.product_name,
            "status": self.status,
            "is_defective": self.is_defective,
            "score": _round_float(self.score),
            "threshold": _round_float(self.threshold),
            "image_size": {
                "width": self.image_size[0],
                "height": self.image_size[1],
            },
            "prompts": list(self.prompts),
            "regions": [region.to_dict() for region in self.regions],
            "source_path": self.source_path,
            "heatmap_path": self.heatmap_path,
            "boxed_path": self.boxed_path,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionResult":
        image_size = data.get("image_size", {})
        if isinstance(image_size, dict):
            size = (
                int(image_size.get("width", 0)),
                int(image_size.get("height", 0)),
            )
        else:
            size = tuple(image_size)  # type: ignore[assignment]

        regions = [DetectionRegion.from_dict(item) for item in data.get("regions", [])]
        is_defective = bool(data.get("is_defective", data.get("status") == "Defective"))
        return cls(
            product_name=str(data.get("product_name", "")),
            is_defective=is_defective,
            score=float(data.get("score", 0)),
            threshold=float(data.get("threshold", 0.5)),
            image_size=size,
            prompts=list(data.get("prompts", [])),
            regions=regions,
            source_path=data.get("source_path"),
            heatmap_path=data.get("heatmap_path"),
            boxed_path=data.get("boxed_path"),
            created_at=str(data.get("created_at", _iso_now())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class BatchItemResult:
    """A result entry for one image in a batch run."""

    image_path: str
    ok: bool
    result: Optional[DetectionResult] = None
    error: Optional[str] = None

    @property
    def is_defective(self) -> bool:
        return bool(self.result and self.result.is_defective)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "ok": self.ok,
            "error": self.error,
            "result": self.result.to_dict() if self.result else None,
        }


@dataclass
class BatchSummary:
    """Aggregated statistics for a batch detection run."""

    product_name: str
    items: List[BatchItemResult]
    started_at: str = field(default_factory=_iso_now)
    finished_at: str = field(default_factory=_iso_now)

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def succeeded(self) -> int:
        return sum(1 for item in self.items if item.ok)

    @property
    def failed(self) -> int:
        return sum(1 for item in self.items if not item.ok)

    @property
    def defective(self) -> int:
        return sum(1 for item in self.items if item.is_defective)

    @property
    def normal(self) -> int:
        return self.succeeded - self.defective

    @property
    def defect_rate(self) -> float:
        if self.succeeded == 0:
            return 0.0
        return self.defective / self.succeeded

    @property
    def average_score(self) -> float:
        scores = [item.result.score for item in self.items if item.result]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def results(self) -> Iterable[DetectionResult]:
        for item in self.items:
            if item.result:
                yield item.result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_name": self.product_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "defective": self.defective,
            "normal": self.normal,
            "defect_rate": _round_float(self.defect_rate),
            "average_score": _round_float(self.average_score),
            "items": [item.to_dict() for item in self.items],
        }


def ensure_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def normalize_extensions(values: Iterable[str]) -> Tuple[str, ...]:
    normalized = []
    for value in values:
        item = value.lower().strip()
        if not item:
            continue
        if not item.startswith("."):
            item = "." + item
        normalized.append(item)
    return tuple(dict.fromkeys(normalized))
