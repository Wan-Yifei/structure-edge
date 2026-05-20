"""
moomoo project — unified entry point

Usage:
  uv run main.py orderflow [orderflow-args...]
  uv run main.py scheduler
"""

import argparse
import importlib

COMMANDS = {
    "orderflow": ("analysis.orderflow", "main"),
    "scheduler": ("analysis.scheduler", "main"),
}


def main():
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
