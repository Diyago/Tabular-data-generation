# -*- coding: utf-8 -*-
"""
Test configuration for tabgan.

We ensure that the project `src` directory is on ``sys.path`` so that both
``src.tabgan`` and sibling top-level packages such as ``_ForestDiffusion``
are importable when running tests from the repository root.
"""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

src_str = os.fspath(SRC_PATH)
if src_str not in sys.path:
    sys.path.insert(0, src_str)

