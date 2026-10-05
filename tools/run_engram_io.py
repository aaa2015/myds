#!/usr/bin/env python3
"""Run CPU Engram lookups on fresh APFS clones; never stop services/use Metal.

Run on the model host during an independently controlled exclusive window.
An APFS clone supplies the same bytes with a distinct file identity. It is a
cold candidate, not proof: retain the phase-bracketed global device counters.
Do not replace clonefile with a normal copy of a several-hundred-GB model.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import uuid


def emit(path, kind, **data):
    row = dict(kind=kind, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **data)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for piece in iter(lambda: f.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def storage():
    return subprocess.check_output(
        ["ioreg", "-r", "-c", "IOBlockStorageDriver", "-l", "-d", "1"],
        text=True, timeout=15)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--tokens-bin", type=Path)
    inputs.add_argument("--text", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--clone-directory", type=Path,
                   help="new directory on the model volume; default beside model")
    p.add_argument("--order", default="1,4,8,8,4,1", help="balanced fresh-clone variants")
    p.add_argument("--warm-runs", type=int, default=2)
    p.add_argument("--mode", choices=["cached", "nocache"], default="cached")
    p.add_argument("--timeout", type=float, default=180)
    args = p.parse_args()
    if sys.platform != "darwin":
        p.error("fresh-clone measurement requires macOS clonefile and ioreg")
    try:
        order = [int(s) for s in args.order.split(",")]
    except ValueError:
        p.error("invalid worker order")
    if not order or any(n not in (1, 4, 8) for n in order) or not 0 <= args.warm_runs <= 7 or args.timeout <= 0:
        p.error("workers must be 1/4/8; warm-runs 0..7; positive timeout")
    model = args.model.resolve(strict=True)
    bench = args.bench.resolve(strict=True)
    fixture = (args.tokens_bin or args.text).resolve(strict=True)
    if not model.is_file() or not fixture.is_file() or not bench.is_file():
        p.error("bench, model and fixture must be regular files")
    args.output.mkdir(parents=True, exist_ok=False)
    window = args.output / "window.jsonl"
    clone_dir = args.clone_directory or model.parent / (".myds-engram-io-" + uuid.uuid4().hex)
    clone_dir.mkdir(parents=False, exist_ok=False)
    lib = ctypes.CDLL(None, use_errno=True)
    clonefile = lib.clonefile
    clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    clonefile.restype = ctypes.c_int
    original = model.stat()
    clones = []
    child = None
    error = None
    env = {k: v for k, v in os.environ.items() if not k.startswith("MYDS_")}
    emit(window, "window_created", scope="CPU production prepare-token path, no Metal",
         model=str(model), model_stat=dict(device=original.st_dev, inode=original.st_ino,
                                          bytes=original.st_size, mtime_ns=original.st_mtime_ns),
         bench_sha256=sha(bench), input_sha256=sha(fixture), fixture=str(fixture),
         input_kind="tokens_bin" if args.tokens_bin else "text", order=order,
         mode=args.mode, runs=1 + args.warm_runs, clone_directory=str(clone_dir),
         boundary="clone identity is only a cold candidate; device counters are global OS storage-driver reads, not per-process or NAND attribution")

    def terminate():
        nonlocal child
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        child = None

    def interrupted(signum, frame):
        raise RuntimeError("interrupted by signal " + str(signum))

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for index, workers in enumerate(order):
            name = "sample-%02d-workers-%d" % (index, workers)
            clone = clone_dir / (name + ".gguf")
            if clonefile(os.fsencode(model), os.fsencode(clone), 0):
                code = ctypes.get_errno()
                raise OSError(code, os.strerror(code), str(clone))
            clones.append(clone)
            stat = clone.stat()
            if stat.st_dev != original.st_dev or stat.st_ino == original.st_ino or stat.st_size != original.st_size:
                raise RuntimeError("clone identity/size mismatch")
            emit(window, "clone_created", name=name, model=str(clone), device=stat.st_dev,
                 inode=stat.st_ino, file_bytes=stat.st_size, workers=workers)
            argv = [str(bench), "--model", str(clone), "--workers", str(workers),
                    "--runs", str(1 + args.warm_runs), "--mode", args.mode, "--gate",
                    "--tokens-bin" if args.tokens_bin else "--text", str(fixture)]
            emit(window, "measurement_start", name=name, argv=argv)
            with (args.output / (name + ".jsonl")).open("w") as log, \
                    (args.output / (name + ".log")).open("w") as stderr:
                child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=stderr, env=env, start_new_session=True)
                # Keep our own binary line buffer: buffered readline can hide
                # lines from select after it consumes several in one read.
                pending = b""
                deadline = time.monotonic() + args.timeout
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(name + " exceeded timeout")
                    if b"\n" not in pending:
                        readable, _, _ = select.select([child.stdout], [], [], min(remaining, 1))
                        if not readable:
                            continue
                        chunk = os.read(child.stdout.fileno(), 65536)
                        if not chunk:
                            if pending:
                                raise RuntimeError(name + " has a truncated JSON record")
                            break
                        pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        row = json.loads(line)
                        log.write(line.decode("utf-8") + "\n")
                        if row["kind"] in ("engram_io_lookup_ready", "engram_io_lookup_done"):
                            log.flush()
                            phase = "storage_before" if row["kind"].endswith("ready") else "storage_after"
                            emit(window, phase, name=name, run=row["run"], ioreg=storage())
                            child.stdin.write(b"\n")
                            child.stdin.flush()
                rc = child.wait(timeout=5)
                child = None
                if rc:
                    raise RuntimeError(name + " exited " + str(rc))
            emit(window, "measurement_end", name=name, returncode=rc)
            clone.unlink()
            clones.remove(clone)
            emit(window, "clone_removed", name=name, model=str(clone))
    except BaseException as exc:
        error = repr(exc)
        emit(window, "window_error", error=error)
    finally:
        terminate()
        for clone in clones:
            clone.unlink(missing_ok=True)
        clone_dir.rmdir()
        current = model.stat()
        unchanged = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) == \
                    (original.st_dev, original.st_ino, original.st_size, original.st_mtime_ns)
        if not unchanged:
            error = error or "source model stat changed"
        emit(window, "window_complete", error=error, source_stat_unchanged=unchanged,
             clones_removed=not clone_dir.exists())
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
