from __future__ import annotations

import argparse
import json
from pathlib import Path

from industrial_ad.batch import BatchDetector
from industrial_ad.config import BatchConfig, DetectorConfig
from industrial_ad.detector import AnomalyDetector
from industrial_ad.reports import write_reports
from industrial_ad.storage import DetectionStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-ad", description="Industrial anomaly detection CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("detect", help="detect one image")
    single.add_argument("image", help="image path")
    single.add_argument("--product", required=True, help="product name, for example bottle")
    single.add_argument("--model-path", default="models/clipseg")
    single.add_argument("--output-dir", default="outputs/single")
    single.add_argument("--threshold", type=float, default=0.72)
    single.add_argument("--json", action="store_true", help="print JSON result")
    single.set_defaults(func=run_detect)

    batch = subparsers.add_parser("batch", help="detect a directory of images")
    batch.add_argument("source", help="image file or directory")
    batch.add_argument("--product", required=True)
    batch.add_argument("--model-path", default="models/clipseg")
    batch.add_argument("--output-dir", default="outputs/batch")
    batch.add_argument("--threshold", type=float, default=0.72)
    batch.add_argument("--no-recursive", action="store_true")
    batch.add_argument("--formats", default="json,csv,html")
    batch.set_defaults(func=run_batch)

    history = subparsers.add_parser("history", help="show local detection history stats")
    history.add_argument("--path", default="records/detection_history_v2.json")
    history.set_defaults(func=run_history)

    return parser


def run_detect(args: argparse.Namespace) -> int:
    config = DetectorConfig(model_path=args.model_path, threshold=args.threshold)
    detector = AnomalyDetector(config=config)
    result = detector.detect_path(args.image, args.product, output_dir=args.output_dir)
    DetectionStore().append(result)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"{result.status}: score={result.score:.4f}, regions={result.region_count}")
        if result.heatmap_path:
            print(f"heatmap: {result.heatmap_path}")
        if result.boxed_path:
            print(f"boxed: {result.boxed_path}")
    return 0


def run_batch(args: argparse.Namespace) -> int:
    config = DetectorConfig(model_path=args.model_path, threshold=args.threshold)
    detector = AnomalyDetector(config=config)
    batch_config = BatchConfig(recursive=not args.no_recursive)
    runner = BatchDetector(detector, batch_config)

    def progress(index: int, total: int, image_path: Path) -> None:
        print(f"[{index}/{total}] {image_path}")

    summary = runner.run(args.source, args.product, args.output_dir, progress)
    DetectionStore().extend(summary.results())
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    written = write_reports(summary, args.output_dir, formats)
    print(
        "done: "
        f"total={summary.total}, defective={summary.defective}, "
        f"failed={summary.failed}, defect_rate={summary.defect_rate:.2%}"
    )
    for path in written:
        print(f"report: {path}")
    return 0


def run_history(args: argparse.Namespace) -> int:
    store = DetectionStore(args.path)
    print(json.dumps(store.stats(), indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
