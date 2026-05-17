#!/usr/bin/env python3
"""兼容入口 → scripts/mpc_track.py"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
runpy.run_path("scripts/mpc_track.py", run_name="__main__")
