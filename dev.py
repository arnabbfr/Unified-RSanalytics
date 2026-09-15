#!/usr/bin/env python3
"""Cross-platform development launcher for Unified-RSanalytics.

    python dev.py             backend (started + healthy) then the desktop UI
    python dev.py backend     backend only
    python dev.py frontend    desktop UI only
    python dev.py stop        stop the backend

Starts the Docker engine itself if it is not already answering. Standard library
only, so it needs nothing beyond the Python that already builds this repo.
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UI_PROJECT = ROOT / "Desktop_App" / "Upgrahan2" / "src" / "GeoSemanticSat.UI" / "GeoSemanticSat.UI.csproj"
SAMPLE_RASTER = ROOT / "data" / "sample_before.tif"
ENGINE_TIMEOUT_SECONDS = 180


class LauncherError(RuntimeError):
    """A step failed in a way the developer has to act on."""


def say(message: str) -> None:
    print(f"==> {message}", flush=True)


def run(command: list[str]) -> None:
    """Run a command with its output attached to this terminal."""
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise LauncherError(f"`{' '.join(command)}` failed with exit code {result.returncode}.")


def require(executable: str, hint: str) -> None:
    if shutil.which(executable) is None:
        raise LauncherError(f"`{executable}` is not on PATH. {hint}")


def docker_engine_is_up() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return result.returncode == 0


def launch_docker_engine() -> None:
    """Best-effort per-platform start of the Docker engine."""
    system = platform.system()

    if system == "Darwin":
        subprocess.Popen(["open", "-a", "Docker"])

    elif system == "Windows":
        import os

        candidates = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "DockerDesktop" / "Docker Desktop.exe",
            Path(os.environ.get("ProgramFiles", "")) / "Docker" / "Docker" / "Docker Desktop.exe",
        ]
        executable = next((path for path in candidates if path.is_file()), None)
        if executable is None:
            raise LauncherError("Docker Desktop was not found. Start it manually, then re-run.")
        subprocess.Popen([str(executable)])

    else:
        # Linux: Docker Desktop ships as a user unit. A plain daemon needs root to
        # start, so ask rather than trigger a sudo password prompt from a launcher.
        started = subprocess.run(
            ["systemctl", "--user", "start", "docker-desktop"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if started.returncode != 0:
            raise LauncherError(
                "The Docker daemon is not running. Start it with "
                "`sudo systemctl start docker`, then re-run."
            )


def ensure_docker_engine() -> None:
    if docker_engine_is_up():
        return

    say("Docker engine is down. Starting it...")
    launch_docker_engine()

    deadline = time.monotonic() + ENGINE_TIMEOUT_SECONDS
    while not docker_engine_is_up():
        if time.monotonic() > deadline:
            raise LauncherError(
                f"The Docker engine did not come up within {ENGINE_TIMEOUT_SECONDS} seconds."
            )
        time.sleep(3)
    print("    Docker engine is up.", flush=True)


def start_backend() -> None:
    require("docker", "Install Docker Desktop, or Docker Engine on Linux.")
    ensure_docker_engine()

    # --wait blocks until the compose healthcheck passes, so everything after this
    # line can assume /health is answering.
    say("Backend: docker compose up -d --wait")
    run(["docker", "compose", "up", "-d", "--wait"])

    # create_all is idempotent, so this is safe on every start.
    say("Database schema")
    run(["docker", "compose", "exec", "-T", "api", "python", "scripts/init_db.py"])

    if not SAMPLE_RASTER.exists():
        say("Seeding sample GeoTIFFs (first run)")
        run(["docker", "compose", "exec", "-T", "api", "python", "scripts/create_sample_data.py"])

    print("    Backend ready -> http://127.0.0.1:8000/docs", flush=True)


def start_frontend() -> None:
    require("dotnet", "Install the .NET 10 SDK.")
    say("Desktop UI: dotnet run")
    run(["dotnet", "run", "--project", str(UI_PROJECT)])


def stop_backend() -> None:
    require("docker", "Install Docker Desktop, or Docker Engine on Linux.")
    run(["docker", "compose", "down"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["all", "backend", "frontend", "stop"],
        help="what to launch (default: all)",
    )
    target = parser.parse_args().target

    actions = {
        "backend": [start_backend],
        "frontend": [start_frontend],
        "stop": [stop_backend],
        "all": [start_backend, start_frontend],
    }

    try:
        for action in actions[target]:
            action()
    except LauncherError as error:
        print(f"\nerror: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
