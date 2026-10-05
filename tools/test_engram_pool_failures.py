#!/usr/bin/env python3
"""Build/run CPU-only Engram fault injection without changing production code.

Only the isolated myds_engram.c object has its pthread/pread/ldexpf names redirected.
The test fixture calls the real library APIs, creates actual worker threads and
checks resource balances, rollback, duplicate reads and retry behavior.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("build/engram-pool-failures"))
    parser.add_argument("--sanitizer", choices=("none", "address", "thread"), default="none")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get("CC", "clang"))
    flags = ["-O1", "-g", "-std=c11", "-Wall", "-Wextra", "-Werror", "-I" + str(source / "include")]
    if args.sanitizer != "none":
        sanitizer = "address,undefined" if args.sanitizer == "address" else "thread"
        flags += ["-fsanitize=" + sanitizer, "-fno-omit-frame-pointer"]
    redirects = ["-D" + name + "=myds_test_" + name for name in (
        "pthread_create", "pthread_join", "pthread_mutex_init", "pthread_mutex_destroy",
        "pthread_cond_init", "pthread_cond_destroy", "pread", "ldexpf",
    )]

    def run(command: list[str]) -> None:
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True, cwd=source)

    objects = []
    for relative, defines in (
        ("src/myds_engram.c", redirects),
        ("src/myds_gguf.c", []),
        ("tests/test_engram_pool_failures.c", []),
    ):
        obj = output / (Path(relative).stem + ".o")
        run(compiler + flags + defines + ["-c", str(source / relative), "-o", str(obj)])
        objects.append(str(obj))
    binary = output / "test_engram_pool_failures"
    run(compiler + flags + objects + ["-lm", "-lpthread", "-o", str(binary)])
    run([str(binary)])


if __name__ == "__main__":
    main()
