"""Install the legacy Fetch runtime locally on Apple Silicon.

Run from roboverify: uv run python3 -m scripts.setup_mujoco_macos
The official framework stays in .runtime/; no /Applications or system libraries
are changed. Adapted from https://github.com/openai/mujoco-py/issues/682.
"""

import argparse
import hashlib
from importlib.util import find_spec
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import urllib.request

URL = "https://github.com/google-deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-macos-universal2.dmg"
SHA256 = "e406c1d2a279ec687a1fb433ca722789ef30f7ef62af9db10ab132c7b5cfd521"


def setup(archive=None):
    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        raise RuntimeError("This helper is for Apple Silicon. Use your platform's mujoco-py setup instead.")
    project = Path(__file__).resolve().parents[1]
    runtime = project / ".runtime/mujoco211"
    framework = runtime / "MuJoCo.framework"
    library = framework / "Versions/A/libmujoco.2.1.1.dylib"
    if not library.is_file():
        with tempfile.TemporaryDirectory(prefix="roboverify-mujoco-") as temporary:
            temporary = Path(temporary)
            if archive is None:
                archive = temporary / "mujoco.dmg"
                print("Downloading official MuJoCo 2.1.1 framework...")
                urllib.request.urlretrieve(URL, archive)
            archive = Path(archive).resolve()
            if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
                raise RuntimeError("MuJoCo download checksum mismatch.")
            mount = temporary / "mount"
            subprocess.run(
                ["hdiutil", "attach", str(archive), "-nobrowse", "-readonly", "-mountpoint", str(mount)],
                check=True,
            )
            try:
                runtime.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    mount / "MuJoCo.app/Contents/Frameworks/MuJoCo.framework",
                    framework,
                    symlinks=True,
                    dirs_exist_ok=True,
                )
            finally:
                subprocess.run(["hdiutil", "detach", str(mount)], check=True)

    # Find the dylib shipped with the Python GLFW package without importing GLFW
    # before mujoco_py (which warns about that import order).
    glfw_spec = find_spec("glfw")
    if glfw_spec is None:
        raise RuntimeError("Run uv sync first to install GLFW and the Python dependencies.")
    glfw_library = Path(glfw_spec.origin).parent / "libglfw.3.dylib"
    if not glfw_library.is_file():
        raise RuntimeError(f"GLFW's macOS library is missing: {glfw_library}")
    (runtime / "bin").mkdir(exist_ok=True)
    links = {
        runtime / "include": framework / "Versions/A/Headers",
        runtime / "bin/libmujoco210.dylib": library,
        runtime / "bin/libglfw.3.dylib": glfw_library,
    }
    for link, target in links.items():
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            raise RuntimeError(f"Refusing to replace non-symlink: {link}")
        link.symlink_to(os.path.relpath(target, link.parent))
    compiler = project / "scripts/mujoco_clang"
    compiler.chmod(compiler.stat().st_mode | 0o111)
    print(f"Installed runtime: {runtime}")
    from synthesis.environment.cee_us_env.runtime import configure_local_mujoco
    configure_local_mujoco()
    from synthesis.environment.cee_us_env.fpp_construction_env import FetchPickAndPlaceConstruction
    print(f"Verified import: {FetchPickAndPlaceConstruction.__name__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded official DMG.")
    setup(parser.parse_args().archive)
