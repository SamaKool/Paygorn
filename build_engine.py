"""
build_engine.py
~~~~~~~~~~~~~~~
Cross-platform build script for the ``hft_auditor`` C++ extension module.

Usage
-----
    python build_engine.py

The script:
  1. Detects the host OS (Linux/WSL or Windows).
  2. Cleans any previous build artefacts.
  3. Runs CMake configure + build inside ``hf auditor/build/``.
  4. Locates the compiled shared library (.so / .pyd) and copies it to the
     project root so that ``import hft_auditor`` works from any script in
     this directory.
"""

import platform
import shutil
import subprocess
import sys
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent          # project root  (fin_auditor/)
SOURCE_DIR = ROOT / "hf auditor"               # note: directory has a space
BUILD_DIR = SOURCE_DIR / "build"

# Glob patterns for the compiled extension in the build tree
SO_PATTERNS = ["hft_auditor*.so", "hft_auditor*.pyd"]

# Destination files that should be removed from the project root on clean
ROOT_ARTEFACTS = list(ROOT.glob("hft_auditor*.so")) + list(ROOT.glob("hft_auditor*.pyd"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run(cmd: list[str], *, description: str = "") -> None:
    """Run *cmd* with real-time stdout/stderr passthrough."""
    if description:
        print(f"\n{'='*60}")
        print(f"  {description}")
        print(f"{'='*60}")

    print(f"$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)   # noqa: S603
    return result


def clean() -> None:
    """Remove existing build artefacts."""
    print("\n── Clean ─────────────────────────────────────────────────────")

    if BUILD_DIR.exists():
        print(f"  Removing build directory: {BUILD_DIR}")
        shutil.rmtree(BUILD_DIR)
    else:
        print(f"  Build directory not found (nothing to remove): {BUILD_DIR}")

    for path in (list(ROOT.glob("hft_auditor*.so")) + list(ROOT.glob("hft_auditor*.pyd"))):
        print(f"  Removing root artefact: {path}")
        path.unlink()

    print("  Clean complete.\n")


def cmake_configure(os_name: str, docker_safe: bool = False) -> None:
    """Run the CMake configure step."""
    cmd = [
        "cmake",
        "-B", str(BUILD_DIR),
        "-S", str(SOURCE_DIR),
        f"-DCMAKE_BUILD_TYPE=Release",
    ]

    if docker_safe:
        # -O1 peaks at ~1.2 GB vs -O3's ~5 GB for this translation unit.
        # DOCKER_SAFE_BUILD=ON triggers the matching CMakeLists.txt option.
        cmd += ["-DDOCKER_SAFE_BUILD=ON", "-DCMAKE_CXX_FLAGS=-O1"]
        print("  Memory-safe mode: -j1, -O1, DOCKER_SAFE_BUILD=ON")

    if os_name == "Windows":
        cmd += ["-G", "Visual Studio 17 2022"]

    run(cmd, description="CMake – Configure")


def cmake_build(docker_safe: bool = False) -> None:
    """Run the CMake build step."""
    cmd = ["cmake", "--build", str(BUILD_DIR), "--config", "Release"]
    if docker_safe:
        # --parallel 1  →  make -j1  →  caps peak RAM to ~1.5 GB
        cmd += ["--parallel", "1"]
    run(cmd, description="CMake – Build")


def copy_to_root() -> Path:
    """
    Walk the build tree looking for the compiled extension and copy it to the
    project root.  Returns the destination path.
    """
    print("\n── Post-Build: Locate & Copy Extension ───────────────────────")

    found: list[Path] = []
    for pattern in SO_PATTERNS:
        found.extend(BUILD_DIR.rglob(pattern))

    if not found:
        sys.exit(
            "ERROR: Could not find a compiled hft_auditor extension under "
            f"{BUILD_DIR}.  Did the build succeed?"
        )

    # If somehow multiple matches exist (e.g. debug + release), prefer the
    # one in a 'Release' sub-directory; otherwise take the first hit.
    chosen = next(
        (p for p in found if "Release" in p.parts or "release" in p.parts),
        found[0],
    )

    dest = ROOT / chosen.name
    print(f"  Source : {chosen}")
    print(f"  Dest   : {dest}")
    shutil.copy2(chosen, dest)
    print(f"  Copied '{chosen.name}' → project root.")
    return dest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build the hft_auditor C++ extension.")
    parser.add_argument(
        "--docker-safe",
        action="store_true",
        help="Memory-safe build for Docker: -j1, -O1 instead of -O3 -march=native.",
    )
    args = parser.parse_args()

    os_name = platform.system()   # 'Linux', 'Windows', or 'Darwin'
    print(f"Detected OS : {os_name}")
    if args.docker_safe:
        print("Mode       : DOCKER_SAFE (--parallel 1, -O1)")

    if os_name not in ("Linux", "Windows"):
        print(
            f"WARNING: OS '{os_name}' is not explicitly supported.  Proceeding "
            "with generic Unix-style CMake commands."
        )

    # 1. Clean
    clean()

    # 2. Configure
    cmake_configure(os_name, docker_safe=args.docker_safe)

    # 3. Build
    cmake_build(docker_safe=args.docker_safe)

    # 4. Copy extension to project root
    dest = copy_to_root()

    print(f"\n✓  Build complete.  Extension available at: {dest}\n")


if __name__ == "__main__":
    main()
