from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Iterable, List

from industrial_ad.types import BatchSummary, DetectionResult


def write_json_report(summary: BatchSummary, path: str | Path) -> Path:
    target = _prepare_target(path)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(), handle, indent=2, ensure_ascii=False)
    return target


def write_csv_report(summary: BatchSummary, path: str | Path) -> Path:
    target = _prepare_target(path)
    with open(target, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "ok",
                "status",
                "score",
                "threshold",
                "region_count",
                "heatmap_path",
                "boxed_path",
                "error",
            ],
        )
        writer.writeheader()
        for item in summary.items:
            result = item.result
            writer.writerow(
                {
                    "image_path": item.image_path,
                    "ok": item.ok,
                    "status": result.status if result else "",
                    "score": f"{result.score:.6f}" if result else "",
                    "threshold": f"{result.threshold:.6f}" if result else "",
                    "region_count": result.region_count if result else "",
                    "heatmap_path": result.heatmap_path if result else "",
                    "boxed_path": result.boxed_path if result else "",
                    "error": item.error or "",
                }
            )
    return target


def write_html_report(summary: BatchSummary, path: str | Path) -> Path:
    target = _prepare_target(path)
    rows = "\n".join(_render_result_row(item.result, item.image_path, item.error) for item in summary.items)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Industrial AD Batch Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    .bad {{ color: #b91c1c; font-weight: 700; }}
    .good {{ color: #047857; font-weight: 700; }}
    .muted {{ color: #6b7280; }}
  </style>
</head>
<body>
  <h1>Industrial AD Batch Report</h1>
  <p>Product: <strong>{html.escape(summary.product_name)}</strong></p>
  <p>
    Total: {summary.total} |
    Succeeded: {summary.succeeded} |
    Failed: {summary.failed} |
    Defective: {summary.defective} |
    Defect rate: {summary.defect_rate:.2%} |
    Average score: {summary.average_score:.4f}
  </p>
  <table>
    <thead>
      <tr>
        <th>Image</th>
        <th>Status</th>
        <th>Score</th>
        <th>Regions</th>
        <th>Heatmap</th>
        <th>Boxed</th>
        <th>Error</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
"""
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(document)
    return target


def write_reports(summary: BatchSummary, output_dir: str | Path, formats: Iterable[str]) -> List[Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        name = fmt.lower().strip()
        if name == "json":
            written.append(write_json_report(summary, target_dir / "report.json"))
        elif name == "csv":
            written.append(write_csv_report(summary, target_dir / "report.csv"))
        elif name == "html":
            written.append(write_html_report(summary, target_dir / "report.html"))
        else:
            raise ValueError(f"unsupported report format: {fmt}")
    return written


def _prepare_target(path: str | Path) -> Path:
    target = Path(path)
    if target.parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _render_result_row(result: DetectionResult | None, image_path: str, error: str | None) -> str:
    image = html.escape(image_path)
    if result is None:
        return f"""
<tr>
  <td>{image}</td>
  <td class="muted">Failed</td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td>{html.escape(error or "")}</td>
</tr>"""

    status_class = "bad" if result.is_defective else "good"
    heatmap = _render_link(result.heatmap_path)
    boxed = _render_link(result.boxed_path)
    return f"""
<tr>
  <td>{image}</td>
  <td class="{status_class}">{html.escape(result.status)}</td>
  <td>{result.score:.6f}</td>
  <td>{result.region_count}</td>
  <td>{heatmap}</td>
  <td>{boxed}</td>
  <td></td>
</tr>"""


def _render_link(path: str | None) -> str:
    if not path:
        return ""
    escaped = html.escape(path)
    return f'<a href="{escaped}">{escaped}</a>'
