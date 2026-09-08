"""Configure the optional project-local Apple Silicon mujoco-py runtime.

MuJoCo 2.1.1's framework can serve the legacy 2.1.0-style directory expected by
mujoco-py. This is a compatibility workaround, not upstream 2.1.0 support:
https://github.com/openai/mujoco-py/issues/682
"""

import os
from pathlib import Path
import platform


def configure_local_mujoco():
    """Call before importing mujoco_py. Existing user settings take precedence."""
    os.environ.pop("LD_PRELOAD", None)
    project_root = Path(__file__).resolve().parents[3]
    runtime = project_root / ".runtime/mujoco211"
    if platform.system() == "Darwin" and platform.machine() == "arm64" and runtime.is_dir():
        os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", str(runtime))
        if Path(os.environ["MUJOCO_PY_MUJOCO_PATH"]).resolve() == runtime:
            os.environ.setdefault("CC", str(project_root / "scripts/mujoco_clang"))
    return Path(os.environ.get("MUJOCO_PY_MUJOCO_PATH", "~/.mujoco/mujoco210")).expanduser()
