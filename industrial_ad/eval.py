from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from industrial_ad.metrics import ClassificationMetrics, evaluate_binary_predictions


TRUE_VALUES = {"1", "true", "yes", "y", "defective", "defect", "anomaly", "bad"}
FALSE_VALUES = {"0", "false", "no", "n", "normal", "good", "ok", "nondefective"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="industrial-ad-eval",
        description="Evaluate binary anomaly detection results with accuracy / precision / recall / F1.",
    )
    parser.add_argument("input", help="CSV or JSON file containing labels and predictions")
    parser.add_argument("--labels", help="Optional CSV/JSON file containing ground-truth labels")
    parser.add_argument("--join-col", default="image_path", help="Column/key used to join input and labels")
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
            return normalize_records([item for item in data if isinstance(item, dict)])
        if isinstance(data, dict):
            if "items" in data and isinstance(data["items"], list):
                return normalize_records([item for item in data["items"] if isinstance(item, dict)])
            return normalize_records([data])
        raise ValueError("JSON input must be an object or a list of objects")

    raise ValueError(f"unsupported input format: {file_format}")


def normalize_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for record in records:
        item = dict(record)
        nested_result = item.get("result")
        if isinstance(nested_result, dict):
            for key, value in nested_result.items():
                item.setdefault(key, value)
        normalized.append(item)
    return normalized


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


def _normalize_join_value(value: Any) -> str:
    text = str(value).strip()
    return Path(text).name if text else text


def merge_labels(
    records: Sequence[Dict[str, Any]],
    labels: Sequence[Dict[str, Any]],
    join_col: str,
    expected_col: str,
) -> List[Dict[str, Any]]:
    label_index = {}
    for label in labels:
        key = _normalize_join_value(_read_column(label, join_col))
        label_index[key] = _read_column(label, expected_col)

    merged = []
    missing = []
    for record in records:
        key = _normalize_join_value(_read_column(record, join_col))
        if key not in label_index:
            missing.append(key)
            continue
        item = dict(record)
        item[expected_col] = label_index[key]
        merged.append(item)

    if missing:
        sample = ", ".join(missing[:5])
        raise KeyError(f"{len(missing)} input rows have no matching label by {join_col}: {sample}")

    return merged


def resolve_predicted_col(records: Sequence[Dict[str, Any]], preferred: str) -> str:
    if records and preferred in records[0]:
        return preferred
    for candidate in ("status", "is_defective", "result", "prediction"):
        if records and candidate in records[0]:
            return candidate
    return preferred


def has_expected_labels(records: Sequence[Dict[str, Any]], expected_col: str) -> bool:
    return bool(records) and expected_col in records[0]


def extract_predictions(
    records: Sequence[Dict[str, Any]],
    predicted_col: str,
    score_col: str,
    threshold: float,
) -> List[bool]:
    predicted = []
    for record in records:
        if predicted_col in record and record[predicted_col] not in (None, ""):
            predicted.append(_parse_bool(record[predicted_col]))
            continue
        score_value = _read_column(record, score_col)
        predicted.append(float(score_value) >= threshold)
    return predicted


def summarize_predictions(predicted: Sequence[bool]) -> Dict[str, Any]:
    total = len(predicted)
    defective = sum(1 for item in predicted if item)
    normal = total - defective
    return {
        "mode": "prediction_summary",
        "total": total,
        "predicted_defective": defective,
        "predicted_normal": normal,
        "predicted_defect_rate": defective / total if total else 0.0,
        "metrics": None,
        "message": "No expected labels were found, so accuracy / precision / recall / F1 cannot be calculated.",
    }


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
    labels_path: str | Path | None = None,
    join_col: str = "image_path",
    expected_col: str = "expected",
    predicted_col: str = "predicted",
    score_col: str = "score",
    threshold: float = 0.72,
    file_format: str = "auto",
) -> ClassificationMetrics:
    records = load_records(input_path, file_format=file_format)
    if labels_path is not None:
        labels = load_records(labels_path, file_format="auto")
        records = merge_labels(records, labels, join_col, expected_col)
    predicted_col = resolve_predicted_col(records, predicted_col)
    expected, predicted = extract_labels(records, expected_col, predicted_col, score_col, threshold)
    return evaluate_binary_predictions(expected, predicted)


def evaluate_or_summarize_file(
    input_path: str | Path,
    labels_path: str | Path | None = None,
    join_col: str = "image_path",
    expected_col: str = "expected",
    predicted_col: str = "predicted",
    score_col: str = "score",
    threshold: float = 0.72,
    file_format: str = "auto",
) -> Dict[str, Any]:
    records = load_records(input_path, file_format=file_format)
    if labels_path is not None:
        labels = load_records(labels_path, file_format="auto")
        records = merge_labels(records, labels, join_col, expected_col)

    predicted_col = resolve_predicted_col(records, predicted_col)
    if not has_expected_labels(records, expected_col):
        predicted = extract_predictions(records, predicted_col, score_col, threshold)
        return summarize_predictions(predicted)

    expected, predicted = extract_labels(records, expected_col, predicted_col, score_col, threshold)
    return {
        "mode": "classification_metrics",
        **evaluate_binary_predictions(expected, predicted).to_dict(),
    }


def format_metrics(metrics: ClassificationMetrics) -> str:
    data = metrics.to_dict()
    return json.dumps(data, indent=2, ensure_ascii=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = evaluate_or_summarize_file(
        args.input,
        labels_path=args.labels,
        join_col=args.join_col,
        expected_col=args.expected_col,
        predicted_col=args.predicted_col,
        score_col=args.score_col,
        threshold=args.threshold,
        file_format=args.format,
    )

    output = json.dumps(result, indent=2, ensure_ascii=False)
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
