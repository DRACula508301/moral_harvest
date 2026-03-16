from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


# Persist per-iteration metrics to JSONL and CSV files.
class IterationResultsWriter:
    # Initialize output paths and CSV writer state.
    def __init__(self, output_dir: Path):
        # Create results directory and concrete output file paths.
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "metrics.jsonl"
        self.csv_path = self.output_dir / "metrics.csv"

        self._jsonl_file = self.jsonl_path.open("w", encoding="utf-8")
        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer: csv.DictWriter[str] | None = None
        self._fieldnames: list[str] | None = None

    # Append one metrics record to both JSONL and CSV files.
    def write(self, row: dict[str, Any]) -> None:
        # Write JSONL record (one JSON object per line).
        self._jsonl_file.write(json.dumps(row, default=str) + "\n")
        self._jsonl_file.flush()

        # Initialize CSV writer on first row and write header.
        if self._csv_writer is None:
            self._fieldnames = list(row.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._csv_writer.writeheader()

        # Fill missing CSV columns with empty values before writing.
        if self._fieldnames is None or self._csv_writer is None:
            return
        normalized_row = {field: row.get(field, "") for field in self._fieldnames}
        self._csv_writer.writerow(normalized_row)
        self._csv_file.flush()

    # Close any open file handles.
    def close(self) -> None:
        self._jsonl_file.close()
        self._csv_file.close()
