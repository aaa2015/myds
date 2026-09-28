#!/usr/bin/env python3
"""Export and optionally build a pinned, isolated ds4 V4.1 reference.

Reads Git objects, never the source repository's working tree. No model is run.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


REVISION = "6f873ad7f5940ab8556abc41ae510853118ecc4a"
ARCHIVE_SHA256 = "bb67197f7444b6c43a045c4b444451c3ef74ce2b9df57b0e5c010f2975198b94"
CORE = ["ds4.o", "ds4_image.o", "ds4_distributed.o", "ds4_tp.o",
        "ds4_ssd.o", "ds4_metal.o", "ds4_layer_pack.o", "ds4_engram.o"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--source-repo", type=Path,
                        help="Git repository containing the pinned commit; dirty files are ignored")
    sources.add_argument("--source-archive", type=Path,
                        help="Previously exported .tar or .tar.gz; verified against the pinned SHA256")
    parser.add_argument("--export-archive", type=Path,
                        help="Write the verified source archive as .tar.gz for offline remote builds")
    parser.add_argument("--output", type=Path, required=True,
                        help="New isolated source/build directory (must not exist)")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error(f"output already exists: {out}")
    if args.source_repo:
        source = args.source_repo.resolve()
        actual = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", REVISION + "^{commit}"], text=True).strip()
        if actual != REVISION:
            parser.error("source revision mismatch")
        names = subprocess.check_output(
            ["git", "-C", str(source), "ls-tree", "-r", "--name-only", REVISION], text=True).splitlines()
        # Keep the Metal CLI and C API closure, all root headers/includes, and image
        # reader dependencies; exclude checkpoints, datasets, docs, binaries, CUDA.
        paths = [name for name in names if (
            "/" not in name and (Path(name).suffix in (".c", ".h", ".m", ".inc")
                                  or name in ("Makefile", "LICENSE")))
            or name.startswith("metal/") or name.startswith("third_party/iris/")]
        archive = subprocess.check_output(
            ["git", "-C", str(source), "archive", "--format=tar", REVISION, "--", *paths])
    else:
        archive = args.source_archive.read_bytes()
        if archive.startswith(b"\x1f\x8b"):
            archive = gzip.decompress(archive)
    if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
        parser.error("source archive SHA256 mismatch; refusing an unpinned reference")
    if args.export_archive:
        args.export_archive.write_bytes(gzip.compress(archive, mtime=0))
    out.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        paths = [member.name for member in tf.getmembers() if member.isfile()]
        for member in tf.getmembers():
            if member.name.startswith("/") or ".." in Path(member.name).parts or member.issym() or member.islnk():
                raise ValueError(f"unexpected archive member: {member.name}")
        tf.extractall(out)
    here = Path(__file__).resolve().parent
    patch = here / "ds4-v41-trace.patch"
    subprocess.run(["patch", "--batch", "-p1", "-i", str(patch)], cwd=out, check=True)
    shutil.copy2(here / "trace_ds4.c", out / "trace_ds4.c")
    shutil.copy2(here / "trace_input.h", out / "trace_input.h")
    manifest = {
        "revision": REVISION,
        "source": "Git objects; working-tree modifications excluded",
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "trace_patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "trace_harness_sha256": hashlib.sha256((out / "trace_ds4.c").read_bytes()).hexdigest(),
        "trace_input_sha256": hashlib.sha256((out / "trace_input.h").read_bytes()).hexdigest(),
        "source_files": len(paths),
        "files": {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in paths},
    }
    (out / "REFERENCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.build:
        subprocess.run(["make", f"-j{args.jobs}", "ds4"], cwd=out, check=True)
        subprocess.run(["ar", "rcs", "libds4.a", *CORE], cwd=out, check=True)
        subprocess.run(["cc", "-O2", "-std=c11", "-I.", "trace_ds4.c", "libds4.a",
                        "-framework", "Foundation", "-framework", "Metal", "-lm", "-pthread",
                        "-o", "trace_ds4"], cwd=out, check=True)
        subprocess.run([str(out / "trace_ds4"), "--help"], cwd=out, check=True)
    print(f"Reference prepared: {out}")
    print(f"Revision: {REVISION}; no model has been loaded or run")


if __name__ == "__main__":
    main()
