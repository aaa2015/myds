#!/usr/bin/env python3
"""Probe /debug/state from inside a window: the 200 path and the loopback gate.

Runs against a server this script starts with MYDS_DEBUG_ENDPOINT=1, so the
production configuration is not modified. Two assertions matter:

  - from 127.0.0.1 with the switch on: 200 and a parsable payload with the three
    top-level sections;
  - from a non-loopback address: 404 even with the switch on, because the server
    has no authentication and this endpoint is the only control.

The LAN probe is issued as a separate process on behalf of the caller (see
--remote-probe), since the check is about the peer address the server observes.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
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


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def emit(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def port_listener(port):
    out = subprocess.run(["lsof", "-nP", "-t", "-a", f"-iTCP:{port}", "-sTCP:LISTEN"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def mem_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    values = {}
    for line in out.splitlines():
        match = re.match(r"Pages (free|inactive|speculative):\s+(\d+)", line)
        if match:
            values[match.group(1)] = int(match.group(2))
    return sum(values.values()) * 16384 / 1e9


def wait_memory(floor_gb, timeout=300.0):
    deadline, seen = time.time() + timeout, []
    while time.time() < deadline:
        available = mem_gb()
        seen.append(round(available, 1))
        if available >= floor_gb:
            return available
        time.sleep(2.0)
    raise RuntimeError(f"memory below {floor_gb} GB: {seen[-5:]}")


def fetch(host, port, timeout=5.0):
    url = f"http://{host}:{port}/debug/state"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:
        return None, str(exc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold-seconds", type=float, default=0.0,
                    help="keep the switch-on server up this long so an external host "
                         "can probe the loopback gate; 0 skips the hold")
    ap.add_argument("--skip-switch-off", action="store_true",
                    help="skip the second (switch-off) server start when a runner "
                         "performs the LAN probe during the hold")
    ap.add_argument("--peer-host", default="0.0.0.0",
                    help="bind address for the test server (default 0.0.0.0 so a LAN "
                         "probe is possible)")
    ap.add_argument("--remote-host", default=None,
                    help="hostname to probe from outside (resolved by the caller)")
    for key, value in DEFAULTS.items():
        ap.add_argument("--" + key.replace("_", "-"), default=value, type=type(value))
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        ap.error("--out already exists")
    if port_listener(a.port) is not None:
        ap.error(f"port {a.port} already in use by {port_listener(a.port)}; "
                 "run inside the window")

    logfile = Path(a.workdir) / "logs" / "local-deepseek-flash.log"
    offset = logfile.stat().st_size if logfile.exists() else 0
    available = wait_memory(a.mem_floor_gb)
    log(f"available {available:.1f} GB")

    env = dict(os.environ)
    env["MYDS_DEBUG_ENDPOINT"] = "1"
    argv = [a.binary, "--config", a.config, "-m", a.model,
            "--port", str(a.port), "--host", a.peer_host, "--ctx-size", "1048576",
            "--alias", a.alias, "--temp", "1.0", "--top-p", "1.0",
            "--max-tokens", "262144", "--thinking", "off", "--vision", a.vision]
    sink = open(logfile, "ab")
    child = subprocess.Popen(argv, cwd=a.workdir, env=env, stdout=sink, stderr=sink,
                             start_new_session=True, close_fds=True)
    sink.close()

    failures = 0
    try:
        started = time.time()
        healthy = False
        while time.time() - started < 480:
            if child.poll() is not None:
                raise RuntimeError(f"server exited {child.returncode}")
            code, _ = fetch("127.0.0.1", a.port, timeout=3)
            if code is not None:
                healthy = True
                break
            time.sleep(2.0)
        if not healthy:
            raise RuntimeError("server never answered")

        # 1) loopback with the switch on must be 200 with every section present
        code, body = fetch("127.0.0.1", a.port)
        ok = code == 200
        sections = {}
        if ok:
            try:
                payload = json.loads(body)
                sections = {k: (k in payload) for k in ("build", "knobs", "counters")}
                ok = all(sections.values())
                identity_ok = (payload.get("build", {}).get("binary_sha256") or "") != ""
                ok = ok and identity_ok
            except ValueError as exc:
                ok = False
                body = f"unparsable: {exc}"
        emit(out, dict(kind="loopback_200", status=code, pass_=ok,
                       sections=sections, body=body[:600]))
        if not ok:
            failures += 1
        log(f"loopback(switch on): {code} sections={sections} -> {'PASS' if ok else 'FAIL'}")

        if a.hold_seconds > 0:
            log(f"holding the switch-on server for {a.hold_seconds:.0f}s "
                f"(external host should probe the LAN gate now)")
            time.sleep(a.hold_seconds)

        if a.skip_switch_off:
            emit(out, dict(kind="note", text="switch-off check skipped by request"))
            log("switch-off check skipped")
            return 0

        # 2) with the switch OFF, loopback must still be 404
        log("restarting with the switch off to re-check the default")
        child.terminate()
        child.wait(timeout=90)
        env_off = dict(env)
        env_off.pop("MYDS_DEBUG_ENDPOINT", None)
        sink = open(logfile, "ab")
        child = subprocess.Popen(argv, cwd=a.workdir, env=env_off, stdout=sink,
                                 stderr=sink, start_new_session=True, close_fds=True)
        sink.close()
        started = time.time()
        while time.time() - started < 480:
            if child.poll() is not None:
                raise RuntimeError(f"server exited {child.returncode}")
            code, _ = fetch("127.0.0.1", a.port, timeout=3)
            if code is not None:
                break
            time.sleep(2.0)
        code, body = fetch("127.0.0.1", a.port)
        ok = code == 404
        emit(out, dict(kind="loopback_404_switch_off", status=code, pass_=ok, body=body[:200]))
        if not ok:
            failures += 1
        log(f"loopback(switch off): {code} -> {'PASS' if ok else 'FAIL'}")

        emit(out, dict(kind="note", text="LAN probe is issued by the caller from another "
                                        "host and appended separately"))
        log("done; LAN probe is the caller's job")
    except BaseException as exc:
        emit(out, dict(kind="abort", error=f"{type(exc).__name__}: {exc}"))
        failures += 1
        log(f"ABORT {type(exc).__name__}: {exc}")
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=90)
            except subprocess.TimeoutExpired:
                log("owned server did not exit; leaving it for the window base")
        settled = wait_memory(a.mem_floor_gb)
        emit(out, dict(kind="teardown", available_gb=round(settled, 1),
                       failures=failures, pass_=failures == 0))
        log(f"teardown: available {settled:.1f} GB, failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
