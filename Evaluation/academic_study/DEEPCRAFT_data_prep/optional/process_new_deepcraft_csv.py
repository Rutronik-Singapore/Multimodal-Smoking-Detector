#!/usr/bin/env python3
"""Prepare future newest-format SEN66 recordings copied into the split folder."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepcraft_csv_common import SPLIT_SOURCE_DIR, run_entry  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(
        run_entry(
            "split",
            "Prepare newest-format SEN66 CSV files from the DEEPCRAFT split folder.",
            SPLIT_SOURCE_DIR,
            {"newer_with_predictions"},
        )
    )
