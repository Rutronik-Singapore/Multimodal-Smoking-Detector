#!/usr/bin/env python3
"""Prepare legacy SEN66 recordings for DEEPCRAFT Studio."""

from deepcraft_csv_common import LEGACY_SOURCE_DIR, run_entry


if __name__ == "__main__":
    raise SystemExit(
        run_entry(
            "legacy",
            "Prepare legacy SEN66 CSV files for DEEPCRAFT Studio.",
            LEGACY_SOURCE_DIR,
            {"legacy"},
        )
    )
