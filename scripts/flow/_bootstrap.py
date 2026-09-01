"""Add repository paths for direct execution of Flow entry points."""

from pathlib import Path
import sys

FLOW_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = FLOW_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent

for path in (REPO_ROOT, SCRIPTS_DIR, FLOW_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
