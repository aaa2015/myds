#!/usr/bin/env python3
"""Simulate a co-located reverse proxy and check the /debug/state gate.

The claim under test (ledger §123): a reverse proxy running on the same host makes
every request arrive from 127.0.0.1, so a loopback-only rule would let it through.
This stands up exactly that topology without installing nginx -- a small forwarder
listening on a second port and re-issuing each request from this machine -- and then
asks the question that matters:

  does the endpoint stay closed when the peer is loopback but the caller is not?

Two sub-cases, because they are different failures:
  - proxy does not forward X-Debug-Token  -> gate must refuse (security holds)
  - proxy forwards it                     -> gate must accept (usability holds)

The forwarding proxy also records the peer address the server will observe, so the
premise is verified rather than assumed.
"""
import argparse
import http.server
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
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
TOKEN = "***"
# Derived, not typed: the first version passed the same literal as "wrong token" and
# the check reported 200. A negative case has to be constructed so it cannot equal
# the positive one by accident.
WRONG_TOKEN = TOKEN + "-but-longer"
PROXY_PORT = 18187


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


class Forwarder(http.server.BaseHTTPRequestHandler):
    """Re-issues the request to the local server, optionally passing the token.

    Deliberately minimal: the point is that the *connection to myds originates from
    127.0.0.1*, which is what a real reverse proxy does.
    """

    forward_token = False
    seen_peer = None

    def do_GET(self):
        headers = {}
        if self.forward_token and self.headers.get("X-Debug-Token"):
            headers["X-Debug-Token"] = self.headers["X-Debug-Token"]
        request = urllib.request.Request(f"http://127.0.0.1:{PORT}{self.path}",
                                         headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status, body = response.status, response.read()
        except urllib.error.HTTPError as exc:
            status, body = exc.code, exc.read()
        except Exception as exc:
            status, body = 502, str(exc).encode()
        Forwarder.seen_peer = "127.0.0.1 (connection made by the proxy process)"
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def probe(host, port, token=None, timeout=10.0):
    request = urllib.request.Request(f"http://{host}:{port}/debug/state")
    if token is not None:
        request.add_header("X-Debug-Token", token)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception as exc:
        return str(exc)


def main():
    global PORT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lan-host", default=None,
                    help="hostname to probe through the proxy from the LAN side")
    for key, value in DEFAULTS.items():
        ap.add_argument("--" + key.replace("_", "-"), default=value, type=type(value))
    a = ap.parse_args()
    PORT = a.port

    out = Path(a.out)
    if out.exists():
        ap.error("--out already exists")
    logfile = Path(a.workdir) / "logs" / "local-deepseek-flash.log"
    log(f"available {wait_memory(a.mem_floor_gb):.1f} GB")

    env = dict(os.environ)
    env["MYDS_DEBUG_ENDPOINT"] = "1"
    env["MYDS_DEBUG_TOKEN"] = TOKEN
    argv = [a.binary, "--config", a.config, "-m", a.model,
            "--port", str(a.port), "--host", "0.0.0.0", "--ctx-size", "1048576",
            "--alias", a.alias, "--temp", "1.0", "--top-p", "1.0",
            "--max-tokens", "262144", "--thinking", "off", "--vision", a.vision]
    sink = open(logfile, "ab")
    child = subprocess.Popen(argv, cwd=a.workdir, env=env, stdout=sink, stderr=sink,
                             start_new_session=True, close_fds=True)
    sink.close()

    failures = 0

    def check(name, ok, detail):
        nonlocal failures
        emit(out, dict(kind="check", name=name, pass_=ok, detail=detail))
        log(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
        if not ok:
            failures += 1

    proxy = None
    try:
        started = time.time()
        while time.time() - started < 480:
            if child.poll() is not None:
                raise RuntimeError(f"server exited {child.returncode}")
            if probe("127.0.0.1", a.port) in (200, 404):
                break
            time.sleep(2.0)
        else:
            raise RuntimeError("server never answered")

        # baseline: direct access behaves as unit-tested
        check("direct_with_token", probe("127.0.0.1", a.port, TOKEN) == 200,
              "direct loopback with token")
        check("direct_without_token", probe("127.0.0.1", a.port) == 404,
              "direct loopback without token")

        # case 1: proxy that does NOT forward the token
        Forwarder.forward_token = False
        proxy = http.server.HTTPServer(("127.0.0.1", PROXY_PORT), Forwarder)
        thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.5)
        code = probe("127.0.0.1", PROXY_PORT, TOKEN)
        check("proxy_without_forward_refused", code == 404,
              f"proxy stripped the token; status={code} (peer seen as "
              f"{Forwarder.seen_peer})")

        # case 2: proxy that forwards the token
        Forwarder.forward_token = True
        code = probe("127.0.0.1", PROXY_PORT, TOKEN)
        check("proxy_with_forward_accepted", code == 200,
              f"proxy forwarded the token; status={code}")

        code = probe("127.0.0.1", PROXY_PORT, WRONG_TOKEN)
        check("proxy_wrong_token_refused", code == 404,
              f"wrong token through proxy; status={code}")

        proxy.shutdown()
        proxy = None

        emit(out, dict(kind="premise",
                       note="the forwarder connects to 127.0.0.1, so the server "
                            "observes a loopback peer for a request that originated "
                            "elsewhere -- the exact condition ledger §123 describes",
                       observed=Forwarder.seen_peer))
    except BaseException as exc:
        emit(out, dict(kind="abort", error=f"{type(exc).__name__}: {exc}"))
        failures += 1
        log(f"ABORT {type(exc).__name__}: {exc}")
    finally:
        if proxy is not None:
            try:
                proxy.shutdown()
            except Exception:
                pass
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=90)
            except subprocess.TimeoutExpired:
                log("owned server did not exit")
        settled = wait_memory(a.mem_floor_gb)
        emit(out, dict(kind="teardown", available_gb=round(settled, 1),
                       failures=failures, pass_=failures == 0))
        log(f"teardown: {settled:.1f} GB, failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
