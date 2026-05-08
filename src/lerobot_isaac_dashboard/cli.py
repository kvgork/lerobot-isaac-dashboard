"""cli.py — Console script wrapper for the lerobot-isaac-dashboard Streamlit app.

Registered as the ``lerobot-isaac-dashboard`` entry-point in pyproject.toml.

Usage::

    lerobot-isaac-dashboard [--workspace PATH] [--session-id ID] [--port PORT]
                            [streamlit_args ...]

Everything not recognised by this parser is passed through to ``streamlit run``
unchanged.  This lets users set e.g. ``--server.headless true`` without any
changes here.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Absolute path to app.py, resolved relative to this file.
_APP_PATH = Path(__file__).parent / "app.py"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lerobot-isaac-dashboard",
        description=(
            "Launch the lerobot-isaac metrics dashboard (Streamlit). "
            "Unknown flags are forwarded to `streamlit run`."
        ),
        add_help=True,
    )
    parser.add_argument(
        "--workspace",
        metavar="PATH",
        default=None,
        help=(
            "Path to the lerobot-isaac workspace root. "
            "Defaults to LEROBOT_ISAAC_WORKSPACE env var or CWD."
        ),
    )
    parser.add_argument(
        "--session-id",
        metavar="ID",
        default=None,
        dest="session_id",
        help="Agent session ID to display. Defaults to the most recent session.",
    )
    parser.add_argument(
        "--port",
        metavar="PORT",
        type=int,
        default=8501,
        help="Port for the Streamlit server (default: 8501).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Console script entrypoint.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]``).  Passing an explicit
        list makes the function testable without launching a real subprocess.

    Returns
    -------
    int
        Exit code from the ``streamlit run`` subprocess (0 = success).
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = _build_parser()
    known, passthrough = parser.parse_known_args(argv)

    # Build the streamlit run command
    cmd: list[str] = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(_APP_PATH),
        "--server.port",
        str(known.port),
    ]

    # Append pass-through streamlit flags BEFORE the '--' separator
    if passthrough:
        cmd.extend(passthrough)

    # Append our own args after '--' so app.py can read them from sys.argv
    app_args: list[str] = []
    if known.workspace:
        app_args.append(f"--workspace={known.workspace}")
    if known.session_id:
        app_args.append(f"--session-id={known.session_id}")

    if app_args:
        cmd.append("--")
        cmd.extend(app_args)

    logger.debug("cli: launching %s", cmd)

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print(
            "Error: streamlit not found. "
            "Install it with: pip install streamlit  (or: pixi install -e dashboard)",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
