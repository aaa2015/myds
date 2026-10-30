#!/usr/bin/env python3
"""E3: does the context-capacity configuration itself cost throughput?

Ledger §105.7 records 32K prefill at 641.3 t/s in one configuration and 616.4 in
another (-3.9%), with decode moving the other way (+4.7% ms/token), and the
difference has never been attributed. The known differences include
`--ctx-size` 524288 -> 1048576 and `MYDS_CHUNK_KEY_MAX` 524288 -> 1048576.

The earlier plan compared those two settings as a single A/B, which cannot
attribute anything: two variables, one arm pair. This tool runs the 2x2 instead:

    arm   ctx        MYDS_CHUNK_KEY_MAX   effective key_comp_max
    A     1048576    1048576              1048576
    B     1048576    524288               524288
    D     524288     524288               524288
    C     524288     1048576              524288   <- clamped, see below

`myds_v41_gpu_scratch_create` clamps `key_max > max_comp` down to `max_comp`
(src/myds_v41_gpu.c), and nothing logs `key_comp_max`, so C is byte-for-byte the
same configuration as B. C is still run as a control: if B and C do not agree,
the clamp reasoning or the arm plumbing is wrong and the run is void.

Arm identity is proven from the outside rather than assumed:
  - ctx is visible in the process argv (`--ctx-size`)
  - MYDS_CHUNK_KEY_MAX is visible in the process environment
The driver records both from the live process, so "the config says X" cannot be
mistaken for "X took effect".

Each boot is a fresh process, and every measured prompt is distinct, so no
measurement can be served from the prefix cache; a row whose server log reports
a cache hit is rejected rather than recorded.

Run it INSIDE an exclusive window (the window base stops production, this runs,
then the base restores). It leaves no myds process behind and waits for memory to
be reclaimed before exiting, so the base's restore is not racing a teardown.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
# Resolved at runtime from --corpus: this script is often staged outside the
# repository (a window harness copies it to /tmp), where parent.parent is not the
# repo root and a module-level constant would silently point at nothing.
CORPUS = HERE.parent / "benchmarks" / "register_only_2026-10-27" / "prompts"

DEFAULTS = dict(
    binary=str(Path.home() / "Projects/myds/bin/myds"),
    config=str(Path.home() / "Projects/myfleet/myds.conf"),
    workdir=str(Path.home() / "Projects/myfleet"),
    model=str(Path.home() / ".lmstudio/models/antirez/deepseek-v4.1-flash-1017/"
              "DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf"),
    vision=str(Path.home() / ".lmstudio/models/antirez/deepseek-v4.1-flash-1017/"
               "DeepSeek-V4.1-Flash-1017-Vision.gguf"),
    port=8187,
    alias="local-deepseek-flash",
    mem_floor_gb=140,
)

# (name, ctx, CHUNK_KEY_MAX). C exists only as a clamp control.
ARMS = [("A", 1048576, "1048576"),
        ("B", 1048576, "524288"),
        ("D", 524288, "524288"),
        ("C", 524288, "1048576")]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def emit(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def pids():
    out = subprocess.run(["pgrep", "-x", "myds"], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def mem_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    values = {}
    for line in out.splitlines():
        match = re.match(r"Pages (free|inactive|speculative):\s+(\d+)", line)
        if match:
            values[match.group(1)] = int(match.group(2))
    return sum(values.values()) * 16384 / 1e9


def total_ram_gb():
    out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()
    if not out.isdigit():
        raise RuntimeError("cannot read hw.memsize")
    return int(out) / float(1 << 30)


def require_capable_host(mem_floor_gb):
    """Refuse to run where the model cannot plausibly fit.

    Starting myds maps a 362 GB file and wires ~150 GB. On a smaller machine that
    is a wired-memory panic, not a failed run, so this is checked before anything
    is stopped rather than after the first allocation.
    """
    total = total_ram_gb()
    if total < mem_floor_gb + 40:
        raise RuntimeError(
            f"this host has {total:.1f} GiB RAM but the run needs >= {mem_floor_gb + 40:.0f} GiB "
            f"(mem floor {mem_floor_gb} GiB plus headroom); refusing to start a 150 GiB model here")
    return total


def stop_all(timeout=90.0):
    for pid in pids():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + timeout
    while time.time() < deadline and pids():
        time.sleep(1.0)
    if pids():
        raise RuntimeError(f"myds did not stop: {pids()}; refusing to SIGKILL")


def wait_memory(floor_gb, timeout=300.0):
    deadline = time.time() + timeout
    saw = []
    while time.time() < deadline:
        available = mem_gb()
        saw.append(round(available, 1))
        if available >= floor_gb:
            return available
        time.sleep(2.0)
    raise RuntimeError(f"available memory below {floor_gb} GB: {saw[-6:]}")


def start(a, ctx, key_max, logfile):
    env = dict(os.environ)
    env.pop("MYDS_INDEXER_REGISTER_ONLY", None)
    env["MYDS_CHUNK_KEY_MAX"] = key_max
    argv = [a.binary, "--config", a.config, "-m", a.model,
            "--port", str(a.port), "--host", "0.0.0.0", "--ctx-size", str(ctx),
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


def observed_identity(port, ctx, key_max):
    """Prove from the live process that the arm was really applied."""
    out = subprocess.run(["lsof", "-nP", "-t", "-a", f"-iTCP:{port}", "-sTCP:LISTEN"],
                         capture_output=True, text=True).stdout.split()
    if not out:
        raise RuntimeError("no listener after start")
    pid = int(out[0])
    argv = subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                          capture_output=True, text=True).stdout
    tokens = argv.split()
    ctx_seen = None
    for i, token in enumerate(tokens):
        if token in ("--ctx-size", "--ctx", "-c") and i + 1 < len(tokens):
            ctx_seen = tokens[i + 1]
    env_text = subprocess.run(["ps", "eww", "-p", str(pid)],
                              capture_output=True, text=True).stdout
    key_seen = None
    for token in env_text.split():
        if token.startswith("MYDS_CHUNK_KEY_MAX="):
            key_seen = token.split("=", 1)[1]
    return {"pid": pid, "ctx_in_argv": ctx_seen, "chunk_key_max_in_env": key_seen,
            "ctx_expected": str(ctx), "chunk_key_max_expected": key_max}


def ask(a, prompt, max_tokens, timeout=1800):
    payload = {"model": a.alias, "temperature": 0, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": prompt}]}
    request = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    return body, time.time() - started


def server_line(logfile, offset):
    with open(logfile, "rb") as handle:
        handle.seek(offset)
        text = handle.read().decode("utf-8", "replace")
    found = None
    for line in text.splitlines():
        if "Completed ChatCompletions" not in line:
            continue
        match = re.search(r"prompt: (\d+) tok, prefill: (\d+) tok in ([\d.]+)s = ([\d.]+) t/s "
                          r"\(cached (\d+)\)", line)
        if match:
            found = dict(prompt_tok=int(match.group(1)), prefill_tok=int(match.group(2)),
                         prefill_s=float(match.group(3)), tps=float(match.group(4)),
                         cached=int(match.group(5)))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="JSONL output path (must not exist)")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--arms", default="A,B,D,C")
    ap.add_argument("--sizes", default="8k,32k")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--decode-tokens", type=int, default=160)
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--corpus", type=Path,
                    default=str(Path.home() / "Projects/myds/benchmarks/"
                                "register_only_2026-10-27/prompts"),
                    help="prompt corpus directory (absolute; the script may be staged)")
    for key, value in DEFAULTS.items():
        ap.add_argument("--" + key.replace("_", "-"), default=value, type=type(value))
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        ap.error("--out already exists; use a fresh path")
    wanted = [arm for arm in a.arms.split(",") if arm]
    unknown = [arm for arm in wanted if arm not in {n for n, _, _ in ARMS}]
    if unknown:
        ap.error(f"unknown arms: {unknown}")
    sizes = [s for s in a.sizes.split(",") if s]
    corpus = a.corpus if a.corpus.is_dir() else CORPUS
    if not corpus.is_dir():
        ap.error(f"corpus directory not found: {a.corpus}")
    needed = [f"d_{size}__v{sample}.txt" for size in sizes for sample in range(a.samples)]
    missing = [name for name in needed + ["d_short__v0.txt"] if not (corpus / name).is_file()]
    if missing:
        ap.error(f"corpus is incomplete, missing {missing} in {corpus}")

    logfile = Path(a.workdir) / "logs" / "local-deepseek-flash.log"
    total = require_capable_host(a.mem_floor_gb)
    log(f"host RAM {total:.1f} GiB, memory floor {a.mem_floor_gb} GB")
    arms = [(name, ctx, key) for name, ctx, key in ARMS if name in wanted]

    emit(out, dict(kind="plan", cycles=a.cycles, arms=[dict(arm=n, ctx=c, chunk_key_max=k)
                                                       for n, c, k in arms],
                   sizes=sizes, samples=a.samples, corpus=str(corpus),
                   note="C is expected to equal B via the key_max>max_comp clamp; "
                        "disagreement between B and C voids the run",
                   binary_sha=subprocess.run(["shasum", "-a", "256", a.binary],
                                             capture_output=True, text=True).stdout.split()[0]))

    try:
        for cycle in range(a.cycles):
            for name, ctx, key in arms:
                log(f"=== cycle {cycle} arm {name} ctx={ctx} key_max={key} ===")
                stop_all()
                available = wait_memory(a.mem_floor_gb)
                log(f"available {available:.1f} GB")
                offset = logfile.stat().st_size if logfile.exists() else 0
                patch = start(a, ctx, key, logfile)
                identity = observed_identity(a.port, ctx, key)
                emit(out, dict(kind="boot", cycle=cycle, arm=name, available_gb=round(available, 1),
                               startup_s=patch, **identity))
                if identity["ctx_in_argv"] != str(ctx) or identity["chunk_key_max_in_env"] != key:
                    raise RuntimeError(f"arm {name} was not applied: {identity}")
                ask(a, "hello", 2)          # warm the process, never timed
                time.sleep(1.0)

                for size in sizes:
                    for sample in range(a.samples):
                        # distinct prompt per sample, so nothing can be cached
                        prompt_path = corpus / f"d_{size}__v{sample % 3}.txt"
                        prompt = prompt_path.read_text()
                        before = logfile.stat().st_size
                        body, wall = ask(a, prompt, 2)
                        time.sleep(0.5)
                        parsed = server_line(logfile, before)
                        if not parsed:
                            raise RuntimeError(f"no server prefill line for {name}/{size}/{sample}")
                        if parsed["cached"] != 0:
                            raise RuntimeError(f"cache hit in a timed run: {parsed}")
                        emit(out, dict(kind="sample", cycle=cycle, arm=name, size=size,
                                       sample=sample, wall_s=round(wall, 3),
                                       prompt_file=prompt_path.name, **parsed))
                        log(f"  {size} v{sample}: {parsed['prefill_tok']} tok in "
                            f"{parsed['prefill_s']}s = {parsed['tps']} t/s")

                if not a.skip_decode:
                    prompt = (corpus / "d_short__v0.txt").read_text()
                    before = logfile.stat().st_size
                    body, wall = ask(a, prompt, a.decode_tokens)
                    usage = body.get("usage", {})
                    completed = usage.get("completion_tokens", 0)
                    emit(out, dict(kind="decode", cycle=cycle, arm=name,
                                   tokens=completed, wall_s=round(wall, 3),
                                   ms_per_token=round(1000.0 * wall / completed, 2)
                                   if completed else None))
                    log(f"  decode {completed} tok in {wall:.2f}s")
                stop_all()
    except BaseException as exc:
        emit(out, dict(kind="abort", error=f"{type(exc).__name__}: {exc}"))
        log(f"ABORT: {type(exc).__name__}: {exc}")
        raise
    finally:
        # Leave nothing running and let the mapping settle, so the outer window's
        # restore is not racing a teardown.
        try:
            stop_all()
            settled = wait_memory(a.mem_floor_gb)
            log(f"left clean; available {settled:.1f} GB")
            emit(out, dict(kind="teardown", available_gb=round(settled, 1)))
        except BaseException as exc:
            log(f"teardown problem: {type(exc).__name__}: {exc}")
            emit(out, dict(kind="teardown_failure", error=str(exc)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
