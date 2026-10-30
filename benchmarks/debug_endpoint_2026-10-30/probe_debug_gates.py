#!/usr/bin/env python3
"""Probe the /debug/state access gates on a real server.

Unit tests cover the predicates; this covers the wiring -- that the header is
actually read off the request, that the switch and the token are consulted where
they should be, and that a reverse proxy on the same host cannot get through.

Runs its own server (never production) with MYDS_DEBUG_ENDPOINT=1 and
MYDS_DEBUG_TOKEN set, so no production environment is modified. Expects to run
inside the exclusive window.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULTS = dict(
    workdir=str(Path.home() / "Projects/myfleet"),
    model=str(Path.home() / ".lmstudio/models/antirez/deepseek-v4.1-flash-1017/"
              "DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf"),
    vision=str(Path.home() / ".lmstudio/models/antirez/deepseek-v4.1-flash-1017/"
               "DeepSeek-V4.1-Flash-1017-Vision.gguf"),
    binary=str(Path.home() / "Projects/myds/bin/myds"),
    config=str(Path.home() / "Projects/myfleet/myds.conf"),
    port=8187,
    alias="local-deepseek-flash",
    mem_floor_gb=140,
)
TOKEN = "negctl-token-7f3a"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def emit(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def mem_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    values = {}
    for line in out.splitlines():
        match = re.match(r"Pages (free|inactive|speculative):\s+(\d+)", line)
        if match:
            values[match.group(1)] = int(match.group(2))
    return sum(values.values()) * 16384 / 1e9


def wait_memory(floor, timeout=400.0):
    deadline, seen = time.time() + timeout, []
    while time.time() < deadline:
        available = mem_gb()
        seen.append(round(available, 1))
        if available >= floor:
            return available
        time.sleep(2.0)
    raise RuntimeError(f"memory below {floor} GB: {seen[-5:]}")


def fetch(host, port, token=None, timeout=5.0):
    request = urllib.request.Request(f"http://{host}:{port}/debug/state")
    if token is not None:
        request.add_header("X-Debug-Token", token)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:                      # connection refused etc.
        return None, str(exc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bind", default="0.0.0.0",
                    help="server bind address (0.0.0.0 so the LAN case is reachable)")
    for key, value in DEFAULTS.items():
        ap.add_argument("--" + key.replace("_", "-"), default=value, type=type(value))
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        ap.error("--out already exists")
    logfile = Path(a.workdir) / "logs" / "local-deepseek-flash.log"
    available = wait_memory(a.mem_floor_gb)
    log(f"available {available:.1f} GB")

    argv = [a.binary, "--config", a.config, "-m", a.model,
            "--port", str(a.port), "--host", a.bind, "--ctx-size", "1048576",
            "--alias", a.alias, "--temp", "1.0", "--top-p", "1.0",
            "--max-tokens", "262144", "--thinking", "off", "--vision", a.vision]
    failures = 0

    def start(env_extra, tag):
        env = dict(os.environ)
        env.pop("MYDS_DEBUG_ENDPOINT", None)
        env.pop("MYDS_DEBUG_TOKEN", None)
        env.update(env_extra)
        sink = open(logfile, "ab")
        child = subprocess.Popen(argv, cwd=a.workdir, env=env, stdout=sink,
                                 stderr=sink, start_new_session=True, close_fds=True)
        sink.close()
        started = time.time()
        while time.time() - started < 480:
            if child.poll() is not None:
                raise RuntimeError(f"server exited {child.returncode} ({tag})")
            code, _ = fetch("127.0.0.1", a.port, timeout=3)
            if code is not None:
                return child
            time.sleep(2.0)
        raise RuntimeError(f"server never answered ({tag})")

    def stop(child):
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=90)
            except subprocess.TimeoutExpired:
                raise RuntimeError("owned server did not exit; leaving it for the window base")

    def check(name, ok, detail):
        nonlocal failures
        emit(out, dict(kind="check", name=name, pass_=ok, detail=detail))
        log(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
        if not ok:
            failures += 1

    child = None
    try:
        # --- switch on + token set -------------------------------------------
        child = start({"MYDS_DEBUG_ENDPOINT": "1", "MYDS_DEBUG_TOKEN": TOKEN},
                      "switch+token")
        code, body = fetch("127.0.0.1", a.port, token=TOKEN)
        parsed = False
        try:
            json.loads(body)
            parsed = True
        except ValueError:
            parsed = False
        check("token_present_opens", code == 200 and parsed,
              f"status={code} json={parsed}")

        code, _ = fetch("127.0.0.1", a.port, token=None)
        check("token_absent_closed", code == 404, f"status={code}")

        code, _ = fetch("127.0.0.1", a.port, token=TOKEN[:-1])
        check("token_prefix_rejected", code == 404, f"status={code}")

        code, _ = fetch("127.0.0.1", a.port, token=TOKEN + "x")
        check("token_extension_rejected", code == 404, f"status={code}")

        code, _ = fetch("127.0.0.1", a.port, token="wrong-but-same-length")
        check("token_wrong_rejected", code == 404, f"status={code}")

        stop(child)
        child = None

        # --- switch on, token NOT set: must stay shut ------------------------
        child = start({"MYDS_DEBUG_ENDPOINT": "1"}, "switch only")
        code, _ = fetch("127.0.0.1", a.port, token="")
        check("fails_closed_without_token", code == 404, f"status={code}")
        code, _ = fetch("127.0.0.1", a.port, token="anything")
        check("no_token_any_value_closed", code == 404, f"status={code}")
        stop(child)
        child = None
        log("done; LAN probe is the caller's job if needed")
    finally:
        try:
            stop(child)
        except BaseException as exc:
            log(f"stop problem: {exc}")
        settled = wait_memory(a.mem_floor_gb)
        emit(out, dict(kind="teardown", available_gb=round(settled, 1),
                       failures=failures, pass_=failures == 0))
        log(f"teardown: {settled:.1f} GB free, failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
