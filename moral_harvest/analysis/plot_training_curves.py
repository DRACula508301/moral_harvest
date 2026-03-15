from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any

import matplotlib.pyplot as plt


# Parse arguments for training-curve plotting.
def parse_args() -> argparse.Namespace:
    # Build parser for metrics input and output controls.
    parser = argparse.ArgumentParser(description="Plot training curves from saved metrics")
    parser.add_argument(
        "--metrics-path",
        required=True,
        help="Path to metrics.jsonl or metrics.csv generated during training",
    )
    parser.add_argument("--x-key", default="iteration")
    parser.add_argument(
        "--y-keys",
        nargs="+",
        default=["episode_reward_mean", "policy_loss", "value_loss", "entropy"],
    )
    parser.add_argument("--output-path", default=None, help="Optional plot output path (.png)")
    parser.add_argument("--title", default="Training Curves")
    return parser.parse_args()


# Convert values to float when possible.
def _to_float(value: Any) -> float | None:
    # Normalize missing values to None.
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Load metric rows from either JSONL or CSV.
def load_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    # Read line-delimited JSON metrics file.
    if metrics_path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    # Read CSV metrics file.
    if metrics_path.suffix.lower() == ".csv":
        with metrics_path.open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    raise ValueError("Unsupported metrics file format. Use .jsonl or .csv")


def resolve_metrics_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.exists():
        return candidate

    windows_style = re.match(r"^(?P<drive>[A-Za-z]):[\\/].+", raw_path)
    if windows_style:
        windows_path = PureWindowsPath(raw_path)
        if windows_path.anchor:
            drive = windows_path.drive[:-1].lower()
            suffix_parts = windows_path.parts[1:]
            wsl_path = Path("/mnt") / drive / Path(*suffix_parts)
            if wsl_path.exists():
                return wsl_path

    return candidate


def build_not_found_error(metrics_path: Path, raw_path: str) -> FileNotFoundError:
    if re.match(r"^[A-Za-z]:[^\\/].+", raw_path):
        return FileNotFoundError(
            "Metrics file not found: "
            f"{metrics_path}\n"
            "Tip: your shell consumed Windows backslashes. In WSL/bash, pass one of:\n"
            "  --metrics-path '/mnt/c/Users/.../metrics.jsonl'\n"
            "  --metrics-path 'C:/Users/.../metrics.jsonl'\n"
            "  --metrics-path 'C:\\\\Users\\\\...\\\\metrics.jsonl'"
        )

    return FileNotFoundError(f"Metrics file not found: {metrics_path}")


# Render and save training curves for selected metrics.
def plot_curves(
    rows: list[dict[str, Any]],
    x_key: str,
    y_keys: list[str],
    output_path: Path,
    title: str,
) -> None:
    # Build x-axis values from metric rows.
    x_values: list[float] = []
    for row in rows:
        x_value = _to_float(row.get(x_key))
        if x_value is not None:
            x_values.append(x_value)

    if not x_values:
        raise ValueError(f"No numeric values found for x-key '{x_key}'.")

    # Create one subplot per metric key.
    fig, axes = plt.subplots(len(y_keys), 1, figsize=(10, 3 * len(y_keys)), sharex=True)
    if len(y_keys) == 1:
        axes = [axes]

    for axis, metric_key in zip(axes, y_keys):
        y_values: list[float] = []
        filtered_x: list[float] = []

        # Collect numeric series for the selected metric.
        for row in rows:
            x_value = _to_float(row.get(x_key))
            y_value = _to_float(row.get(metric_key))
            if x_value is None or y_value is None:
                continue
            filtered_x.append(x_value)
            y_values.append(y_value)

        if y_values:
            axis.plot(filtered_x, y_values, label=metric_key)
            axis.set_ylabel(metric_key)
            axis.grid(True, alpha=0.3)
            axis.legend(loc="best")
        else:
            axis.text(0.5, 0.5, f"No numeric data for {metric_key}", ha="center", va="center")
            axis.set_ylabel(metric_key)

    axes[-1].set_xlabel(x_key)
    fig.suptitle(title)
    fig.tight_layout()

    # Ensure parent directories exist before saving plot.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# Entrypoint for curve plotting from persisted training metrics.
def main() -> None:
    # Parse args and validate metrics file path.
    args = parse_args()
    metrics_path = resolve_metrics_path(args.metrics_path)
    if not metrics_path.exists():
        raise build_not_found_error(metrics_path, args.metrics_path)

    # Load rows and determine output image path.
    rows = load_metrics(metrics_path)
    if not rows:
        raise ValueError("Metrics file contains no rows.")

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else metrics_path.parent / "training_curves.png"
    )

    # Render and persist plot.
    plot_curves(
        rows=rows,
        x_key=args.x_key,
        y_keys=args.y_keys,
        output_path=output_path,
        title=args.title,
    )

    print(f"plot_saved={output_path}")


if __name__ == "__main__":
    main()
