#!/usr/bin/env python3
"""E2: decompose a warm append -- the cost of adding tokens to an existing context.

Ledger §0.1 and the mask-skip screen both show that appending ~2K tokens to a
~98K context costs far more than the marginal prefill rate predicts. This driver
reproduces that request shape over HTTP and splits it with
MYDS_REQUEST_PHASES=1, plus whatever the engine's own profile switches report.

The shape matters:
  request 1 (cold)  base prompt, unique nonce so nothing can be reused
  request 2 (warm)  base + append text, same process

The warm request reuses the prefix and only prefills the appended tail. If
cached_tokens is not close to the base length, the server refused the reuse and
the cell measures a full prefill -- so every row records cached_tokens and the
comparison is refused when reuse did not happen.

Why this is not one of the existing harnesses: bench_indexer_mask_warm.m drives
the engine directly and therefore cannot show the server-side split (render,
setup, snapshot restore). The gap this is meant to attribute lives partly outside
the engine, so it has to be measured through the request path.

Run inside an exclusive window. Requires MYDS_REQUEST_PHASES=1 on the server
being measured.
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
    corpus=str(Path.home() / "Projects/myds/benchmarks/"
               "register_only_2026-10-27/prompts"),
)

PHASES = re.compile(r"request_phases (\{.*\})")
COMPLETED = re.compile(
    r"prompt: (\d+) tok, prefill: (\d+) tok in ([\d.]+)s = ([\d.]+) t/s \(cached (\d+)\)")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def emit(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def mem_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    values = {}
    for line in out.splitlines():
        match = re.match(r"Pages (free|inactive|speculative):\s+(\d+)", line)
        if match:
            values[match.group(1)] = int(match.group(2))
    return sum(values.values()) * 16384 / 1e9


def total_ram_gb():
    raw = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()
    if not raw.isdigit():
        raise RuntimeError("cannot read hw.memsize")
    return int(raw) / float(1 << 30)


def port_listener(port):
    """PID listening on `port`, or None. Used only to refuse unsafe starts."""
    out = subprocess.run(["lsof", "-nP", "-t", "-a", f"-iTCP:{port}", "-sTCP:LISTEN"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


class OwnedServer:
    """A server this driver started, and only that server.

    Deliberately does not search for myds by name: `pgrep -x myds` would also
    match the production process, and a teardown that kills work it did not start
    is how a measurement turns into an outage. This driver ran once without a
    window and left production down exactly that way.
    """

    def __init__(self):
        self.child = None
        self.port = None

    def start(self, a, logfile, extra_env):
        if port_listener(a.port) is not None:
            raise RuntimeError(
                f"port {a.port} is already in use by pid {port_listener(a.port)}; "
                "this driver only starts its own server, so production must be stopped "
                "first (run it inside the exclusive window)")
        env = dict(os.environ)
        env.pop("MYDS_INDEXER_REGISTER_ONLY", None)
        env["MYDS_REQUEST_PHASES"] = "1"
        for key, value in extra_env.items():
            env[key] = value
        argv = [a.binary, "--config", a.config, "-m", a.model,
                "--port", str(a.port), "--host", "0.0.0.0", "--ctx-size", "1048576",
                "--alias", a.alias, "--temp", "1.0", "--top-p", "1.0",
                "--max-tokens", "262144", "--thinking", "off", "--vision", a.vision]
        sink = open(logfile, "ab")
        self.child = subprocess.Popen(argv, cwd=a.workdir, env=env, stdout=sink,
                                      stderr=sink, start_new_session=True, close_fds=True)
        sink.close()
        self.port = a.port
        started = time.time()
        while time.time() - started < 480:
            if self.child.poll() is not None:
                raise RuntimeError(f"server exited during startup with {self.child.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=3) as r:
                    if r.status == 200:
                        return round(time.time() - started, 1)
            except Exception:
                pass
            time.sleep(2.0)
        raise RuntimeError("server did not become healthy")

    def stop(self, timeout=90.0):
        if self.child is None:
            return
        if self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"owned server {self.child.pid} did not exit; refusing to SIGKILL "
                    "(a killed 362 GB mapping leaves the kernel tearing down while the "
                    "next process pages in)")
        self.child = None


def wait_memory(floor_gb, timeout=300.0):
    deadline, seen = time.time() + timeout, []
    while time.time() < deadline:
        available = mem_gb()
        seen.append(round(available, 1))
        if available >= floor_gb:
            return available
        time.sleep(2.0)
    raise RuntimeError(f"available memory below {floor_gb} GB: {seen[-6:]}")


def start(a, logfile, extra_env):
    env = dict(os.environ)
    env.pop("MYDS_INDEXER_REGISTER_ONLY", None)
    # The whole point of E2 is the split, so it must be on.
    env["MYDS_REQUEST_PHASES"] = "1"
    for key, value in extra_env.items():
        env[key] = value
    argv = [a.binary, "--config", a.config, "-m", a.model,
            "--port", str(a.port), "--host", "0.0.0.0", "--ctx-size", "1048576",
            "--alias", a.alias, "--temp", "1.0", "--top-p", "1.0",
            "--max-tokens", "262144", "--thinking", "off", "--vision", a.vision]
    with open(logfile, "ab") as sink:
        subprocess.Popen(argv, cwd=a.workdir, env=env, stdout=sink, stderr=sink,
                         start_new_session=True, close_fds=True)
    started = time.time()
    while time.time() - started < 480:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=3) as r:
                if r.status == 200:
                    return round(time.time() - started, 1)
        except Exception:
            pass
        time.sleep(2.0)
    raise RuntimeError("service did not become healthy")


def ask(a, prompt, max_tokens=2, timeout=3600):
    payload = {"model": a.alias, "temperature": 0, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": prompt}]}
    request = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    return body, time.time() - started


def read_new(logfile, offset):
    with open(logfile, "rb") as handle:
        handle.seek(offset)
        return handle.read().decode("utf-8", "replace")


def parse_phases(text):
    found = None
    for line in text.splitlines():
        match = PHASES.search(line)
        if match:
            try:
                found = json.loads(match.group(1))
            except ValueError:
                found = {"unparsable": match.group(1)}
    return found


def parse_completed(text):
    found = None
    for line in text.splitlines():
        if "Completed ChatCompletions" not in line:
            continue
        match = COMPLETED.search(line)
        if match:
            found = dict(prompt_tok=int(match.group(1)), prefill_tok=int(match.group(2)),
                         prefill_s=float(match.group(3)), tps=float(match.group(4)),
                         cached=int(match.group(5)))
    return found


def build_base(corpus, byte_budget, marker):
    """Concatenate distinct corpus files to exactly `byte_budget` bytes.

    Budget is bytes, not tokens: this corpus mixes ASCII and CJK, so a bytes-per-
    token constant is wrong by more than 2x depending on which slice is taken. The
    real token count is read back from the server log and recorded per row.
    """
    budget = int(byte_budget)
    data = bytearray()
    for name in ("d_32k__v0.txt", "d_32k__v1.txt", "d_32k__v2.txt",
                 "d_8k__v0.txt", "d_8k__v1.txt", "d_8k__v2.txt"):
        path = corpus / name
        if not path.is_file():
            continue
        data += path.read_bytes()
        if len(data) >= budget:
            break
    if not data:
        raise RuntimeError(f"no corpus material in {corpus}")
    # Work in bytes throughout: this corpus is mixed ASCII/CJK, so len(str) is
    # not the byte count and a character-based budget silently produced a body
    # 40% short of the requested size. Trim on a UTF-8 boundary so the request
    # body stays valid.
    data = data[:budget]
    while data and (data[-1] & 0xC0) == 0x80:
        data.pop()
    if data and data[-1] >= 0xC0:
        data.pop()
    return marker + data.decode("utf-8")


APPEND_UNIT_BYTES = 4.2   # ASCII-heavy filler: measured on this string below


def build_append(tokens_wanted, nonce):
    """Distinct filler; its first bytes differ from the base, so this is a real
    append and the prefix still matches up to the join point."""
    unit = f"[{nonce}] alpha beta gamma delta 0123456789 tail "
    needed = int(tokens_wanted * APPEND_UNIT_BYTES)
    return unit * (needed // len(unit) + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="8k")
    ap.add_argument("--appends", default="512")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--server-env", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="extra environment for the server this driver starts "
                         "(e.g. --server-env MYDS_PROFILE_MOE=1 to split prefill)")
    ap.add_argument("--cold-only", action="store_true",
                    help="measure cold prefill only; useful for the short-prompt "
                         "fixed-cost decomposition (E1)")
    ap.add_argument("--reuse-server", action="store_true",
                    help="measure an already-running server instead of booting per repetition")
    for key, value in DEFAULTS.items():
        ap.add_argument("--" + key.replace("_", "-"), default=value, type=type(value))
    a = ap.parse_args()

    a.corpus = Path(a.corpus)
    out = Path(a.out)
    if out.exists():
        ap.error("--out already exists; use a fresh path")
    if not a.corpus.is_dir():
        ap.error(f"corpus not found: {a.corpus}")
    ram = total_ram_gb()
    if ram < a.mem_floor_gb + 40:
        ap.error(f"this host has {ram:.1f} GiB RAM; needs >= {a.mem_floor_gb + 40:.0f}")

    # Fail on the real precondition first. Waiting out the memory floor only to
    # then discover that production is still listening wastes five minutes and
    # reports the wrong reason.
    if not a.reuse_server:
        holder = port_listener(a.port)
        if holder is not None:
            ap.error(f"port {a.port} is in use by pid {holder}: stop production first "
                     "(this driver only ever starts and stops its own server)")

    sizes = [s for s in a.sizes.split(",") if s]
    appends = [] if a.cold_only else [int(x) for x in a.appends.split(",") if x]
    logfile = Path(a.workdir) / "logs" / "local-deepseek-flash.log"

    emit(out, dict(kind="plan", sizes=sizes, appends=appends, reps=a.reps,
                   note="warm row is valid only if cached_tok ~= base tokens",
                   binary_sha=subprocess.run(["shasum", "-a", "256", a.binary],
                                             capture_output=True, text=True).stdout.split()[0],
                   corpus=str(a.corpus)))

    owned = OwnedServer()
    try:
        if not a.reuse_server:
            available = wait_memory(a.mem_floor_gb)
            extra = {}
            for item in a.server_env:
                if "=" not in item:
                    ap.error(f"--server-env needs KEY=VALUE, got {item!r}")
                key, value = item.split("=", 1)
                extra[key] = value
            startup = owned.start(a, logfile, extra)
            emit(out, dict(kind="boot", startup_s=startup,
                           available_gb=round(available, 1)))
            log(f"booted in {startup}s")
        for rep in range(a.reps):
            for size in sizes:
                # Sizes are byte budgets in KiB. The marker makes each size a
                # distinct prefix, otherwise the larger prompt would simply extend
                # the smaller one and "cold" would be a cache hit.
                marker = f"[run{rep}-{size}-{int(time.time()) % 100000}] "
                base = build_base(a.corpus, int(size.rstrip("k")) * 1024, marker)
                nonce = f"r{rep}s{size}{int(time.time()) % 100000}"
                # cold: base only
                before = logfile.stat().st_size
                _, cold_wall = ask(a, base)
                time.sleep(0.5)
                cold_text = read_new(logfile, before)
                cold_done, cold_phases = parse_completed(cold_text), parse_phases(cold_text)
                emit(out, dict(kind="cold", rep=rep, size=size, wall_s=round(cold_wall, 3),
                               bytes=len(base), completed=cold_done, phases=cold_phases))
                log(f"  cold: {cold_done['prefill_tok'] if cold_done else '?'} tok in "
                    f"{cold_done['prefill_s'] if cold_done else '?'}s")
                base_tokens = cold_done["prompt_tok"] if cold_done else 0

                for append_tokens in appends:
                    tail = build_append(append_tokens, nonce)
                    prompt = base + tail
                    before = logfile.stat().st_size
                    _, warm_wall = ask(a, prompt)
                    time.sleep(0.5)
                    text = read_new(logfile, before)
                    done, phases = parse_completed(text), parse_phases(text)
                    reused = done["cached"] if done else 0
                    ratio = (reused / base_tokens) if base_tokens else 0.0
                    emit(out, dict(kind="warm", rep=rep, size=size, append=append_tokens,
                                   wall_s=round(warm_wall, 3), bytes=len(prompt),
                                   completed=done, phases=phases,
                                   base_tokens=base_tokens, reused_tokens=reused,
                                   reuse_ratio=round(ratio, 4)))
                    log(f"  warm +{append_tokens}: prefill {done['prefill_tok'] if done else '?'} tok "
                        f"in {done['prefill_s'] if done else '?'}s, reused {reused}/{base_tokens} "
                        f"({ratio * 100:.1f}%)")
                    time.sleep(1.0)
    except BaseException as exc:
        emit(out, dict(kind="abort", error=f"{type(exc).__name__}: {exc}"))
        log(f"ABORT: {type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            owned.stop()
            settled = wait_memory(a.mem_floor_gb)
            emit(out, dict(kind="teardown", available_gb=round(settled, 1)))
            log(f"left clean; available {settled:.1f} GB")
        except BaseException as exc:
            log(f"teardown problem: {exc}")
            emit(out, dict(kind="teardown_failure", error=str(exc)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
