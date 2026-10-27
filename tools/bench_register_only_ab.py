#!/usr/bin/env python3
"""A/B the register-only indexer on a real myds server, with the measurement
discipline that ledger §105 requires.

Design points, each one a lesson from an earlier bad run:

  * Interleaved arms (O1 N1 O2 N2 O3 N3), not blocked. Blocked order lets thermal
    or page-cache drift land entirely on one arm.
  * 3 samples per boot x 3 boots per arm = 9 timed samples per arm, so a single
    outlier cannot move the conclusion.
  * Every prompt is a distinct corpus file, so the server's prefix cache cannot
    serve it. Any row with cached != 0 is recorded and excluded from statistics.
  * The first request after each boot uses the SAME file (the "fixed" sample),
    when the cache is provably empty; its output is compared byte-for-byte
    between arms. That is the equality check, text-level only.
  * GPU telemetry is sampled *concurrently* with the run. Sampling afterwards
    only proves the GPU went idle.
  * Counters are read before and after each boot, so "the arm actually took
    effect" is evidence (encoded) rather than an assumption.
  * The deployed default is restored at the end, even if a step fails.

Output schema matches the historical file (rig.jsonl) so the same verifier can
read both.

Usage:
    python3 tools/bench_register_only_ab.py --out /tmp/rig_run.jsonl
    python3 tools/verify_register_only_bench.py --ab /tmp/rig_run.jsonl
"""
import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "benchmarks" / "register_only_2026-10-27" / "prompts"
DEFAULTS = dict(
    binary=str(Path.home() / "Projects/myds/bin/myds"),
    config=str(Path.home() / "Projects/myfleet/myds.conf"),
    workdir=str(Path.home() / "Projects/myfleet"),
    log=str(Path.home() / "Projects/myfleet/logs/local-deepseek-flash.log"),
    model=str(Path.home() / ".lmstudio/models/antirez/deepseek-v4.1-flash-1017/"
              "DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf"),
    vision=str(Path.home() / ".lmstudio/models/antirez/deepseek-v4.1-flash-1017/"
               "DeepSeek-V4.1-Flash-1017-Vision.gguf"),
    port=8187,
    alias="local-deepseek-flash",
    ctx=1048576,
    case="p32k_ab",
    mem_floor_gb=110,
    timed_per_boot=3,
    boots=3,
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def emit(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def pids():
    r = subprocess.run(["pgrep", "-x", "myds"], capture_output=True, text=True)
    return [int(x) for x in r.stdout.split()]


def mem_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    v = {}
    for line in out.splitlines():
        m = re.match(r"Pages (free|inactive|speculative):\s+(\d+)", line)
        if m:
            v[m.group(1)] = int(m.group(2))
    return (v.get("free", 0) + v.get("inactive", 0) + v.get("speculative", 0)) * 16384 / 1e9


def stop():
    for p in pids():
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for _ in range(150):
        if not pids():
            return
        time.sleep(0.2)
    for p in pids():
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for _ in range(150):
        if not pids():
            return
        time.sleep(0.2)


def start(a, env_value):
    env = dict(os.environ)
    env.pop("MYDS_INDEXER_REGISTER_ONLY", None)
    env.pop("MYDS_INDEXER_REGISTER_ONLY_STRICT", None)
    if env_value is not None:
        env["MYDS_INDEXER_REGISTER_ONLY"] = env_value
    argv = [a.binary, "--config", a.config, "-m", a.model,
            "--port", str(a.port), "--host", "0.0.0.0", "--ctx-size", str(a.ctx),
            "--alias", a.alias, "--temp", "1.0", "--top-p", "1.0",
            "--max-tokens", "262144", "--thinking", "off", "--vision", a.vision]
    with open(a.log, "ab") as f:
        subprocess.Popen(argv, cwd=a.workdir, env=env, stdout=f, stderr=f,
                         start_new_session=True, close_fds=True)
    base = f"http://127.0.0.1:{a.port}"
    t0 = time.time()
    while time.time() - t0 < 420:
        try:
            with urllib.request.urlopen(base + "/health", timeout=3) as r:
                if r.status == 200:
                    return True, round(time.time() - t0, 1)
        except Exception:
            pass
        time.sleep(2)
    return False, None


def ask(a, prompt, max_tokens, timeout=1800):
    payload = {"model": a.alias, "temperature": 0, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d, time.time() - t0


def server_line(logfile, offset):
    with open(logfile, "rb") as f:
        f.seek(offset)
        txt = f.read().decode("utf-8", "replace")
    best = None
    for line in txt.splitlines():
        if "Completed ChatCompletions" not in line:
            continue
        m = re.search(r"prompt: (\d+) tok, prefill: (\d+) tok in ([\d.]+)s = ([\d.]+) t/s "
                      r"\(cached (\d+)\)", line)
        if m:
            best = dict(prompt_tok=int(m.group(1)), prefill_tok=int(m.group(2)),
                        prefill_s=float(m.group(3)), tps=float(m.group(4)),
                        cached=int(m.group(5)))
    return best


def parse_gpu(text):
    blocks, cur = [], None
    for line in text.splitlines():
        m = re.search(r"GPU HW active frequency:\s+(\d+)", line)
        if m:
            cur = {"mhz": int(m.group(1))}
            blocks.append(cur)
            continue
        m = re.search(r"GPU HW active residency:\s+([\d.]+)%", line)
        if m and cur is not None:
            cur["residency"] = float(m.group(1))
            continue
        m = re.search(r"GPU Power:\s+(\d+)", line)
        if m and cur is not None:
            cur["power_mw"] = int(m.group(1))
    return [b for b in blocks if "residency" in b and "power_mw" in b]


def counters(pid):
    """Read the process atomics. Requires passwordless sudo for lldb."""
    if not pid:
        return None
    try:
        r = subprocess.run(["sudo", "lldb", "-b", "-p", str(pid),
                            "-o", "expr -l c++ -- (unsigned long long)malloc(64)",
                            "-o", "detach", "-o", "quit"],
                           capture_output=True, text=True, timeout=180)
        m = re.search(r"\(unsigned long long\) \$0 = (\d+)", r.stdout)
        if not m:
            return None
        buf = m.group(1)
        r = subprocess.run(["sudo", "lldb", "-b", "-p", str(pid),
                            "-o", f"expr -l c++ -- (void)myds_v41_gpu_indexer_register_only_counters((void*){buf}, 0)",
                            "-o", f"memory read -c 6 -f d -s 8 {buf}",
                            "-o", "detach", "-o", "quit"],
                           capture_output=True, text=True, timeout=180)
        v = [int(x) for x in re.findall(r"^0x[0-9a-f]+: (\d+)$", r.stdout, re.M)]
        if len(v) == 6:
            return dict(requests=v[0], encoded=v[1], fallbacks=v[2],
                        failures=v[3], f32=v[4], bf16=v[5])
    except Exception as e:
        log(f"counter read failed: {e}")
    return None


def run_boot(a, out, arm, env_value, boot):
    log(f"=== boot {arm}{boot} (MYDS_INDEXER_REGISTER_ONLY={env_value}) ===")
    stop()
    for _ in range(180):
        if mem_gb() >= a.mem_floor_gb:
            break
        time.sleep(2)
    gb = round(mem_gb(), 1)
    log(f"available {gb} GB, myds procs {len(pids())}")
    ok, secs = start(a, env_value)
    if not ok:
        emit(out, dict(kind="boot_failure", arm=arm, boot=boot))
        return False
    pid = pids()[0] if pids() else None
    before = counters(pid) if a.counters else None
    emit(out, dict(kind="boot", arm=arm, boot=boot, env=env_value, mem_gb=gb,
                   startup_s=secs, counters_before=before, pid=[pid] if pid else []))
    log(f"healthy in {secs}s pid={pid}")

    warm = (CORPUS / f"{a.case}__v{boot % 3}.txt").read_text()
    ask(a, warm, 2)
    time.sleep(2)

    # the fixed sample first: cache is empty, so this is the equality check.
    # It must be a file that no timed run uses, otherwise it warms the cache for
    # the first timed run and that run performs no prefill.
    tasks = [("fixed", (CORPUS / "ab_fixed.txt").read_text())]
    for i in range(a.timed_per_boot):
        tasks.append(("nonce", (CORPUS / f"{a.case}__v{i}.txt").read_text()))
    if len({p for _, p in tasks}) != len(tasks):
        log("FATAL: duplicate prompt inside one boot; the later run would hit the cache")
        emit(out, dict(kind="abort", reason="duplicate_prompt_in_boot", arm=arm, boot=boot))
        return False

    for kind, prompt in tasks:
        off = os.path.getsize(a.log)
        pm = None
        if a.gpu:
            pm = subprocess.Popen(["sudo", "powermetrics", "-n", "58", "-i", "1000",
                                   "--samplers", "gpu_power"],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            time.sleep(1.2)
        d, wall = ask(a, prompt, 2)
        gpu = {}
        if pm is not None:
            try:
                gpu = parse_gpu(pm.communicate(timeout=120)[0])
            except subprocess.TimeoutExpired:
                pm.kill()
        time.sleep(0.6)
        p = server_line(a.log, off)
        text = d["choices"][0]["message"]["content"]

        # A row without a server prefill line, or one that reports a cache hit, did
        # not measure prefill. Refusing it here is the whole point: an earlier run
        # recorded such rows as blanks and they silently entered the average.
        if not p:
            log(f"  ABORT: {kind} produced no server prefill line (wall={wall:.2f}s). "
                f"Most likely the prompt was served from cache.")
            emit(out, dict(kind="abort", reason="no_server_line", arm=arm, boot=boot,
                           sample=kind, wall_s=round(wall, 3)))
            return False
        if p.get("cached"):
            log(f"  ABORT: {kind} was served from prefix cache (cached={p['cached']}); "
                f"no prefill happened")
            emit(out, dict(kind="abort", reason="cache_hit", arm=arm, boot=boot,
                           sample=kind, cached=p["cached"]))
            return False

        rec = dict(kind="run", arm=arm, boot=boot, env=env_value, sample=kind,
                   sha=hashlib.sha256(text.encode()).hexdigest()[:16],
                   completion_tok=d["usage"]["completion_tokens"],
                   prompt_usage=d["usage"]["prompt_tokens"],
                   wall_s=round(wall, 3), text=text, **p)
        if gpu:
            res = [b["residency"] for b in gpu]
            mhz = [b["mhz"] for b in gpu]
            pw = [b["power_mw"] for b in gpu]
            rec["gpu_detail"] = dict(samples=len(gpu),
                                     res_median=sorted(res)[len(res) // 2],
                                     res_max=max(res), mhz_modes=sorted(set(mhz)),
                                     power_median_mw=sorted(pw)[len(pw) // 2],
                                     power_max_mw=max(pw))
        emit(out, rec)
        log(f"  {kind}: {p['prefill_s']}s = {p['tps']} t/s cached={p['cached']} "
            f"wall={wall:.2f}s" + (f" gpu={rec['gpu_detail']['res_median']}% "
                                   f"{rec['gpu_detail']['power_median_mw']/1000:.1f}W"
                                   if gpu else ""))
        time.sleep(1.2)

    after = counters(pid) if a.counters else None
    emit(out, dict(kind="boot_end", arm=arm, boot=boot, counters_after=after,
                   delta=None if not (before and after) else
                   {k: after[k] - before[k] for k in after}))
    return True


def preflight(a):
    """Fail fast and loudly, before anything is restarted.

    Every check here exists because the alternative is worse: spending seven
    minutes before failing, or starting a model that does not fit in RAM and
    taking the machine down with a wired-memory panic.
    """
    problems = []
    for label, p in (("binary", a.binary), ("config", a.config),
                     ("model", a.model), ("vision", a.vision), ("workdir", a.workdir)):
        if not Path(p).exists():
            problems.append(f"{label} not found: {p}")
    if not CORPUS.exists():
        problems.append(f"corpus not found: {CORPUS} (run make corpus-check / "
                        f"tools/make_register_only_corpus.py)")
    else:
        need = [f"{a.case}__v{i}.txt" for i in range(3)] + ["ab_fixed.txt"]
        for f in dict.fromkeys(need):
            if not (CORPUS / f).exists():
                problems.append(f"corpus file missing: {f}")

    # Total RAM, not free RAM: this benchmark pins ~150 GB and a smaller machine
    # dies by wired-memory panic rather than by OOM kill.
    total_gb = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                  capture_output=True, text=True).stdout or 0) / 2**30
    if total_gb < 180:
        problems.append(f"this machine has {total_gb:.0f} GiB RAM; the benchmark pins ~150 GB "
                        f"and a 64/96/128 GiB host can panic from wired exhaustion. "
                        f"Refusing to start. Run this on the 192 GiB machine that holds the weights.")

    if a.gpu or a.counters:
        r = subprocess.run(["sudo", "-n", "true"], capture_output=True)
        if r.returncode != 0:
            what = []
            if a.counters:
                what.append("process counters (lldb)")
            if a.gpu:
                what.append("GPU telemetry (powermetrics)")
            problems.append("passwordless sudo is required for " + " and ".join(what) +
                            "; without it the arm-separation check cannot run, so the "
                            "result would not be verifiable. Configure it, or accept that "
                            "the run proves nothing about which kernel executed.")

    if problems:
        print("PREFLIGHT FAILED — nothing was started, no service was touched:", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        return False
    log(f"preflight ok: {total_gb:.0f} GiB RAM, corpus {len(list(CORPUS.glob('*.txt')))} files")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="JSONL output path")
    ap.add_argument("--case", default=DEFAULTS["case"])
    ap.add_argument("--boots", type=int, default=DEFAULTS["boots"])
    ap.add_argument("--arms", default="OFF,ON",
                    help="which arms to run, comma separated (default OFF,ON). "
                         "A single arm is useful for smoke-testing the harness.")
    ap.add_argument("--timed-per-boot", type=int, default=DEFAULTS["timed_per_boot"])
    ap.add_argument("--no-gpu", action="store_true", help="skip powermetrics sampling")
    ap.add_argument("--no-counters", action="store_true", help="skip atomic counter reads")
    ap.add_argument("--no-restore", action="store_true", help="leave the A/B arm env in place")
    ap.add_argument("--skip-preflight", action="store_true", help="bypass the safety checks (not advised)")
    for k, v in DEFAULTS.items():
        if k in ("case", "boots", "timed_per_boot"):
            continue
        ap.add_argument("--" + k.replace("_", "-"), default=v, type=type(v))
    a = ap.parse_args()
    a.gpu = not a.no_gpu
    a.counters = not a.no_counters

    if not a.skip_preflight and not preflight(a):
        return 2
    if len(pids()) > 1:
        sys.exit(f"{len(pids())} myds processes already running; refusing to start another")

    open(a.out, "w").close()
    arms = [x.strip().upper() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in arms if x not in ("OFF", "ON")]
    if bad:
        sys.exit(f"unknown arm(s): {bad}; use OFF and/or ON")
    ok = True
    try:
        for b in range(1, a.boots + 1):
            # interleave: OFF then ON within the same boot index
            for arm in arms:
                envv = "0" if arm == "OFF" else "1"
                if not run_boot(a, a.out, arm, envv, b):
                    ok = False
    finally:
        if not a.no_restore:
            log("restoring deployed default (no env override => default-on)")
            stop()
            for _ in range(180):
                if mem_gb() >= a.mem_floor_gb:
                    break
                time.sleep(2)
            restored, _ = start(a, None)
            c = counters(pids()[0] if pids() else None) if a.counters else None
            emit(a.out, dict(kind="restore", ok=restored, pid=pids(), counters=c))
            log(f"restored={restored} pid={pids()}")
    print(f"\nwrote {a.out}")
    print("now run: python3 tools/verify_register_only_bench.py --ab " + a.out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
