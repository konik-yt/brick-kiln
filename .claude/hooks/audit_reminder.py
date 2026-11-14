"""PostToolUse hook: after an edit to split, sampling, baseline, training or evaluation code,
remind Claude to run the leakage-auditor subagent before committing.

Reads the hook JSON on stdin. Prints nothing for other files.
"""

import json
import sys
from pathlib import PurePath

WATCHED = {
    "src/02_extra_patches.py",
    "src/03_split.py",
    "src/04_baselines.py",
    "src/05_train.py",
    "src/06_evaluate.py",
    "src/config.py",
}

try:
    data = json.load(sys.stdin)
except ValueError:
    sys.exit(0)

path = (data.get("tool_input") or {}).get("file_path") or (data.get("tool_response") or {}).get("filePath") or ""
parts = PurePath(path.replace("\\", "/")).parts
rel = "/".join(parts[-2:]) if len(parts) >= 2 else ""

if rel in WATCHED:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": (
                f"{rel} affects the district split, sampling, baselines, training data or evaluation. "
                "Before committing, run the leakage-auditor subagent on the uncommitted changes "
                "and fix every FAIL it reports."
            ),
        }
    }))
