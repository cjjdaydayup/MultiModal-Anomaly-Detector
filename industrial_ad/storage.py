from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from industrial_ad.types import DetectionResult


class DetectionStore:
    """JSON-backed store for detection history."""

    def __init__(self, path: str | Path = "records/detection_history_v2.json") -> None:
        self.path = Path(path)

    def append(self, result: DetectionResult) -> None:
        data = self.load_raw()
        data.append(result.to_dict())
        self._write(data)

    def extend(self, results: Iterable[DetectionResult]) -> None:
        data = self.load_raw()
        data.extend(result.to_dict() for result in results)
        self._write(data)

    def load(self) -> List[DetectionResult]:
        return [DetectionResult.from_dict(item) for item in self.load_raw()]

    def load_raw(self) -> List[dict]:
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"history file must contain a list: {self.path}")
        return data

    def search(
        self,
        product_name: Optional[str] = None,
        defective: Optional[bool] = None,
        min_score: Optional[float] = None,
    ) -> List[DetectionResult]:
        results = self.load()
        filtered = []
        for result in results:
            if product_name and result.product_name != product_name:
                continue
            if defective is not None and result.is_defective != defective:
                continue
            if min_score is not None and result.score < min_score:
                continue
            filtered.append(result)
        return filtered

    def stats(self) -> Dict[str, float | int]:
        results = self.load()
        total = len(results)
        defective = sum(1 for item in results if item.is_defective)
        average_score = sum(item.score for item in results) / total if total else 0.0
        return {
            "total": total,
            "defective": defective,
            "normal": total - defective,
            "defect_rate": defective / total if total else 0.0,
            "average_score": average_score,
        }

    def clear(self) -> None:
        self._write([])

    def _write(self, data: List[dict]) -> None:
        if self.path.parent:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
