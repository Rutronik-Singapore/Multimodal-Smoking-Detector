#!/usr/bin/env python3
"""Prepare PSE84 SEN66 recordings for DEEPCRAFT Studio."""

from deepcraft_csv_common import PSE84_SOURCE_DIR, run_entry


if __name__ == "__main__":
    raise SystemExit(
        run_entry(
            "pse84",
            "Prepare PSE84 SEN66 CSV files for DEEPCRAFT Studio.",
            PSE84_SOURCE_DIR,
            {"newer_without_predictions", "newer_with_predictions"},
        )
    )
