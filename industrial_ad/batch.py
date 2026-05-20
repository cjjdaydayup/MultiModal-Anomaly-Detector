from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional

from industrial_ad.config import BatchConfig
from industrial_ad.detector import AnomalyDetector
from industrial_ad.types import BatchItemResult, BatchSummary, normalize_extensions

ProgressCallback = Callable[[int, int, Path], None]


def discover_images(
    root: str | Path,
    extensions: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    recursive: bool = True,
) -> List[Path]:
    """Return image paths under a file or directory."""

    source = Path(root)
    normalized = normalize_extensions(extensions)
    if source.is_file():
        return [source] if source.suffix.lower() in normalized else []
    if not source.exists():
        raise FileNotFoundError(f"image source does not exist: {source}")

    pattern = "**/*" if recursive else "*"
    images = [
        path
        for path in source.glob(pattern)
        if path.is_file() and path.suffix.lower() in normalized
    ]
    return sorted(images)


class BatchDetector:
    """Run anomaly detection over a folder of images."""

    def __init__(
        self,
        detector: AnomalyDetector,
        config: Optional[BatchConfig] = None,
    ) -> None:
        self.detector = detector
        self.config = config or BatchConfig()

    def iter_images(self, source: str | Path) -> Iterator[Path]:
        for image_path in discover_images(
            source,
            self.config.normalized_extensions(),
            self.config.recursive,
        ):
            yield image_path

    def run(
        self,
        source: str | Path,
        product_name: str,
        output_dir: str | Path = "outputs/batch",
        progress: Optional[ProgressCallback] = None,
    ) -> BatchSummary:
        images = list(self.iter_images(source))
        started_at = datetime.now().replace(microsecond=0).isoformat()
        items: List[BatchItemResult] = []
        total = len(images)

        for index, image_path in enumerate(images, start=1):
            if progress:
                progress(index, total, image_path)
            try:
                result = self.detector.detect_path(
                    image_path,
                    product_name,
                    output_dir=Path(output_dir) / "visuals",
                )
                items.append(BatchItemResult(str(image_path), ok=True, result=result))
            except Exception as exc:
                items.append(BatchItemResult(str(image_path), ok=False, error=str(exc)))
                if not self.config.continue_on_error:
                    raise

        return BatchSummary(
            product_name=product_name,
            items=items,
            started_at=started_at,
            finished_at=datetime.now().replace(microsecond=0).isoformat(),
        )
