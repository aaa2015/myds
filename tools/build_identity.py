#!/usr/bin/env python3
"""Report the toolchain identity that a build's reproducibility depends on.

The ledger used to say a deployed binary was "byte-identical to macmini". That
claim decayed silently: the two machines disagreed on both the clang version and
the SDK while the sentence stayed unchanged. Measured on 2026-10-30 (§117):

    clang 2100.3.34.2 + SDK 27.0  ->  972cdce2...   (macmini, Xcode 27.1 default)
    clang 2100.4.1.1  + SDK 27.2  ->  6ce62f86...   (macstudio, and macmini via CLT 27.2)

So an identity claim is undecidable unless it states both versions. This prints
them, and can fail when they differ from a recorded pair, so the check is a
command instead of a memory.

Usage:
  build_identity.py                       print the pair
  build_identity.py --expect-clang X --expect-sdk Y
                                          exit 1 when either differs
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "build-identity.json"


def run(argv, env=None):
    try:
        return subprocess.run(argv, capture_output=True, text=True, env=env).stdout
    except FileNotFoundError:
        return ""


def sdk_version_from_plist(path):
    """Authoritative SDK version, read from the SDK the compiler will use.

    `xcrun --show-sdk-version` without `--sdk` currently fails on macmini because
    it resolves a CommandLineTools SDK path that xcodebuild declines to open, even
    though the SDK itself is intact (SDKSettings.plist says 27.2). Reading the
    plist avoids depending on that quirk and reports the version that is actually
    on disk.
    """
    if not path:
        return None
    plist = Path(path) / "SDKSettings.plist"
    if not plist.is_file():
        return None
    out = run(["/usr/libexec/PlistBuddy", "-c", "Print :Version", str(plist)]).strip()
    return out or None


def probe(developer_dir=None):
    env = dict(os.environ)
    if developer_dir:
        env["DEVELOPER_DIR"] = developer_dir
    clang = run(["clang", "--version"], env=env).splitlines()
    version = clang[0].strip() if clang else ""
    match = re.search(r"clang-(\d+\.\d+\.\d+(?:\.\d+)?)", version)
    path = run(["xcrun", "--sdk", "macosx", "--show-sdk-path"], env=env).strip() or None
    sdk = sdk_version_from_plist(path) or \
        (run(["xcrun", "--sdk", "macosx", "--show-sdk-version"], env=env).strip() or None)
    return {
        "clang_banner": version,
        "clang_version": match.group(1) if match else None,
        "sdk_version": sdk,
        "sdk_path": path,
        "developer_dir": env.get("DEVELOPER_DIR") or run(["xcode-select", "-p"], env=env).strip(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expect-clang", help="required clang version, e.g. 2100.4.1.1")
    ap.add_argument("--expect-sdk", help="required SDK version, e.g. 27.2")
    ap.add_argument("--developer-dir", help="probe a specific DEVELOPER_DIR")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    ap.add_argument("--record", action="store_true",
                    help=f"write the probed pair to {RECORD.relative_to(ROOT)}")
    a = ap.parse_args()

    info = probe(a.developer_dir)
    info["host"] = os.uname().nodename

    if a.record:
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        RECORD.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
        print(f"recorded {RECORD.relative_to(ROOT)}", file=sys.stderr)

    if a.json:
        print(json.dumps(info, indent=2, sort_keys=True))
    else:
        print(f"host      : {info['host']}")
        print(f"developer : {info['developer_dir']}")
        print(f"clang     : {info['clang_version']}   ({info['clang_banner']})")
        print(f"sdk       : {info['sdk_version']}   {info['sdk_path']}")

    problems = []
    if a.expect_clang and info["clang_version"] != a.expect_clang:
        problems.append(f"clang {info['clang_version']} != expected {a.expect_clang}")
    if a.expect_sdk and info["sdk_version"] != a.expect_sdk:
        problems.append(f"sdk {info['sdk_version']} != expected {a.expect_sdk}")
    if problems:
        for item in problems:
            print(f"MISMATCH {item}", file=sys.stderr)
        print("VERDICT: FAIL -- a byte-identity claim cannot be made from this "
              "toolchain (see ledger §117)", file=sys.stderr)
        return 1
    if a.expect_clang or a.expect_sdk:
        print("VERDICT: PASS -- toolchain matches the recorded pair")
    return 0


if __name__ == "__main__":
    sys.exit(main())
