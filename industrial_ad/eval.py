from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from industrial_ad.metrics import ClassificationMetrics, evaluate_binary_predictions, threshold_scores


TRUE_VALUES = {"1", "true", "yes", "y", "defective", "defect", "anomaly", "bad"}
FALSE_VALUES = {"0", "false", "no", "n", "normal", "good", "ok", "nondefective"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="industrial-ad-eval",
        description="Evaluate binary anomaly detection results with accuracy / precision / recall / F1.",
    )
    parser.add_argument("input", help="CSV or JSON file containing labels and predictions")
    parser.add_argument("--expected-col", default="expected", help="Ground-truth label column/key")
    parser.add_argument("--predicted-col", default="predicted", help="Predicted label column/key")
    parser.add_argument("--score-col", default="score", help="Score column/key used with threshold")
    parser.add_argument("--threshold", type=float, default=0.72, help="Threshold used when predicted label is absent")
    parser.add_argument(
        "--format",
        choices=["auto", "csv", "json"],
        default="auto",
        help="Force input format instead of detecting from file extension",
    )
    parser.add_argument("--output", help="Optional path to save metrics as JSON")
    return parser


def load_records(path: str | Path, file_format: str = "auto") -> List[Dict[str, Any]]:
    target = Path(path)
    if file_format == "auto":
        file_format = target.suffix.lower().lstrip(".")

    if file_format == "csv":
        with open(target, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader)

    if file_format == "json":
        with open(target, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            if "items" in data and isinstance(data["items"], list):
                return [item for item in data["items"] if isinstance(item, dict)]
            return [data]
        raise ValueError("JSON input must be an object or a list of objects")

    raise ValueError(f"unsupported input format: {file_format}")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ValueError(f"cannot parse boolean value: {value!r}")


def _read_column(record: Dict[str, Any], column: str) -> Any:
    if column not in record:
        raise KeyError(f"missing column/key: {column}")
    return record[column]


def extract_labels(
    records: Sequence[Dict[str, Any]],
    expected_col: str,
    predicted_col: str,
    score_col: str,
    threshold: float,
) -> Tuple[List[bool], List[bool]]:
    expected: List[bool] = []
    predicted: List[bool] = []

    for record in records:
        expected_value = _read_column(record, expected_col)
        expected.append(_parse_bool(expected_value))

        if predicted_col in record and record[predicted_col] not in (None, ""):
            predicted.append(_parse_bool(record[predicted_col]))
            continue

        score_value = _read_column(record, score_col)
        predicted.append(float(score_value) >= threshold)

    return expected, predicted


def evaluate_file(
    input_path: str | Path,
    expected_col: str = "expected",
    predicted_col: str = "predicted",
    score_col: str = "score",
    threshold: float = 0.72,
    file_format: str = "auto",
) -> ClassificationMetrics:
    records = load_records(input_path, file_format=file_format)
    expected, predicted = extract_labels(records, expected_col, predicted_col, score_col, threshold)
    return evaluate_binary_predictions(expected, predicted)


def format_metrics(metrics: ClassificationMetrics) -> str:
    data = metrics.to_dict()
    return json.dumps(data, indent=2, ensure_ascii=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    metrics = evaluate_file(
        args.input,
        expected_col=args.expected_col,
        predicted_col=args.predicted_col,
        score_col=args.score_col,
        threshold=args.threshold,
        file_format=args.format,
    )

    output = format_metrics(metrics)
    print(output)

    if args.output:
        target = Path(args.output)
        if target.parent:
            target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
