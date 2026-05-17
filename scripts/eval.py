#!/usr/bin/env python3
"""量化评估入口。等价于原 ``eval_koopman.py``。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from koopman.paths import setup_repo

setup_repo()

from koopman.evalkit import main

if __name__ == "__main__":
    raise SystemExit(main())
