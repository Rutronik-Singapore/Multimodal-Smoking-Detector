#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dt_pipeline_common import parse_args, run_dt0

if __name__ == "__main__":
    run_dt0(parse_args("DT0"))
