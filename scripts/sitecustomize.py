"""Make the repository root and flat script modules importable for direct CLI runs."""

from pathlib import Path
import sys

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

for path in (REPO_ROOT, SCRIPTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
