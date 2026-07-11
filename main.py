"""
moomoo project — unified entry point

Usage:
  uv run main.py trade_viewer_qt [args...]   # PyQtGraph viewer (current)
  uv run main.py trade_viewer    [args...]   # Matplotlib viewer (legacy)
  uv run main.py scheduler
  uv run main.py scanner                     # SMC signal scanner
  uv run main.py fvg_backscan --symbol US.SOXL --start 2026-05-01   # FVG-watch backscan
  uv run main.py replay_trainer               # K-line replay trainer (paper trading practice)
"""

import argparse
import importlib

COMMANDS = {
    "trade_viewer_qt": ("analysis.trade_viewer_qt", "main"),  # PyQtGraph (current)
    "trade_viewer":    ("analysis.trade_viewer",    "main"),  # Matplotlib (legacy)
    "scheduler":       ("analysis.scheduler",       "main"),
    "scanner":         ("analysis.signal_scanner",  "main"),
    "fvg_backscan":    ("analysis.fvg_backscan",    "main"),
    "replay_trainer":  ("analysis.replay_trainer",  "main"),
}


def main():
    """Parse the subcommand and delegate to the appropriate module's main()."""
    p = argparse.ArgumentParser(
        prog="main.py",
        description="moomoo toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k}" for k in COMMANDS),
    )
    p.add_argument("command", choices=COMMANDS.keys(), help="subcommand to run")
    args, rest = p.parse_known_args()

    module_path, fn_name = COMMANDS[args.command]
    mod = importlib.import_module(module_path)
    getattr(mod, fn_name)(rest)


if __name__ == "__main__":
    main()
