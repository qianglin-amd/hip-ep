#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Run oga-server straight from a source checkout, with no install step.

    python product/serve.py                 # http://127.0.0.1:11434
    python product/serve.py --port 8080 --engine replay

`pip install -e product` gives you the `oga-server` and `hipep` commands instead; this
launcher exists so that cloning the repo and running one command is enough to see the
product working.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for package_dir in (HERE / "oga-server", HERE / "toolkit"):
    if package_dir.is_dir():
        sys.path.insert(0, str(package_dir))

from hipep_server.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
