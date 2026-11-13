"""Start OpenD, wait until its API answers, then start the scheduler.

The order is not a preference. The collectors the scheduler launches connect
to OpenD's API port on startup, so a scheduler that comes up first finds
nothing to talk to. Doing it by hand means remembering that every time the
machine reboots -- and this machine rebooted seven times in thirteen hours,
each one costing every minute of tape until somebody noticed.

What this does NOT do is log in to OpenD. OpenD asks for that interactively
unless its own config carries the credentials, which is a decision about
where a trading password lives and is not one to make inside a launcher. If
login is still pending, the wait below times out and says so, in one line,
instead of leaving a scheduler running against a dead port.

Both programs draw windows, so this has to run inside a logged-in desktop
session. A scheduled task set to "run whether user is logged on or not"
puts them in session 0 where neither can show a window.

    uv run python scripts/start_stack.py           # start what is missing
    uv run python scripts/start_stack.py --check   # report, change nothing
"""
from __future__ import annotations

import argparse
import pathlib
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
OPEND = (pathlib.Path.home() / "AppData" / "Roaming" / "moomoo_OpenD"
         / "moomoo_OpenD.exe")
OPEND_XML = OPEND.parent / "OpenD.xml"
LOG = ROOT / "logs" / "analysis" / "start_stack.log"

# How long to wait for OpenD's API to answer. Generous: the port opens only
# after login, and a person typing a password is the slow part.
API_WAIT_S = 300
POLL_S = 2.0


def say(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def api_endpoint() -> tuple[str, int]:
    """Where OpenD serves, read from its own config rather than assumed."""
    host, port = "127.0.0.1", 11111
    try:
        root = ET.parse(OPEND_XML).getroot()
        for tag, cast in (("ip", str), ("api_port", int)):
            el = root.find(f".//{tag}")
            if el is not None and (el.text or "").strip():
                if tag == "ip":
                    host = el.text.strip()
                else:
                    port = cast(el.text.strip())
    except Exception as exc:                                # noqa: BLE001
        say(f"could not read {OPEND_XML.name} ({exc}); using {host}:{port}")
    return host, port


def api_up(host: str, port: int) -> bool:
    """True when something accepts a connection on the API port.

    A TCP accept is the readiness signal, not the process existing: OpenD
    runs for as long as it sits on its login screen, and the port is not
    open until that is done.
    """
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except OSError:
        return False


def running(name_fragment: str) -> int | None:
    import psutil

    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        cl = " ".join(p.info["cmdline"] or [])
        if name_fragment.lower() in cl.lower():
            return p.info["pid"]
    return None


def scheduler_pid() -> int | None:
    """The interpreter running the scheduler, past the uv trampoline.

    `.venv\\Scripts\\python.exe` execs the real interpreter, so a naive
    search finds the wrapper and reports a scheduler that is not there.
    """
    import psutil

    for p in psutil.process_iter(["pid", "cmdline"]):
        cl = " ".join(p.info["cmdline"] or [])
        if "main.py" in cl and "scheduler" in cl and ".venv" not in cl \
                and "nohup" not in cl and "uv.exe run" not in cl:
            return p.info["pid"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report what is up and exit, starting nothing")
    ap.add_argument("--wait", type=int, default=API_WAIT_S,
                    help=f"seconds to wait for the API (default {API_WAIT_S})")
    args = ap.parse_args()

    host, port = api_endpoint()
    opend = running("moomoo_OpenD.exe")
    sched = scheduler_pid()
    up = api_up(host, port)

    say(f"OpenD process : {opend or 'not running'}")
    say(f"OpenD API     : {host}:{port} {'answering' if up else 'closed'}")
    say(f"scheduler     : {sched or 'not running'}")
    if args.check:
        return 0 if (up and sched) else 1

    # ── 1. OpenD ─────────────────────────────────────────────────────────
    if opend is None:
        if not OPEND.exists():
            say(f"OpenD not found at {OPEND}")
            return 2
        say("starting OpenD")
        subprocess.Popen([str(OPEND)], cwd=str(OPEND.parent))
    elif not up:
        say("OpenD is running but its API is closed -- it is most likely "
            "still on the login screen")

    # ── 2. wait for the API, which is what the collectors need ───────────
    if not up:
        deadline = time.monotonic() + args.wait
        said = False
        while time.monotonic() < deadline:
            if api_up(host, port):
                up = True
                break
            if not said and time.monotonic() > deadline - args.wait + 20:
                say(f"waiting for {host}:{port} -- log in to OpenD if it is "
                    f"asking")
                said = True
            time.sleep(POLL_S)
        if not up:
            say(f"gave up after {args.wait}s: the API never answered. NOT "
                f"starting the scheduler -- its collectors would come up "
                f"against a dead port and churn.")
            return 3
        say("OpenD API is answering")

    # ── 3. scheduler ─────────────────────────────────────────────────────
    if sched is not None:
        say(f"scheduler already running (pid {sched}); leaving it alone")
        return 0
    say("starting the scheduler")
    subprocess.Popen([sys.executable, "main.py", "scheduler"], cwd=str(ROOT))
    for _ in range(30):
        time.sleep(1.0)
        pid = scheduler_pid()
        if pid:
            say(f"scheduler up (pid {pid})")
            return 0
    say("the scheduler did not appear within 30s")
    return 4


if __name__ == "__main__":
    sys.exit(main())
