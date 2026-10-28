#!/usr/bin/env python3
"""Build an isolated long-context register-only correctness probe.

The existing qualified shader, production runtime and frozen fixture are never
modified. The derived candidate differs only in its two uniform upper guards;
this script also derives a separate long-shape fixture. By default this builds
only. Explicit --run submits GPU work and must be used in an exclusive window.
Only SHA-256 fingerprints of fixture buffers are stored, not the raw tensors.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
APPROVED = "a62922c6f5937236bd0e354dd06943b39bfa5f788419c16a8700da793769d616"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_one(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one source anchor: {old!r}")
    return text.replace(old, new, 1)


def require_idle():
    rows = subprocess.check_output(["ps", "-axo", "pid=,comm="], text=True)
    busy = []
    for row in rows.splitlines():
        fields = row.strip().split(None, 1)
        if len(fields) != 2:
            continue
        name = Path(fields[1]).name.lower()
        if name in {"myds", "myglm", "ds4", "llama-server", "ollama"} or name.startswith("bench_"):
            busy.append(row.strip())
    if busy:
        raise RuntimeError("GPU isolation precheck found inference/benchmark processes: " + "; ".join(busy))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="new private directory outside repository")
    parser.add_argument("--run", action="store_true", help="explicitly run GPU correctness tests after compile")
    parser.add_argument("--deployed-metallib", type=Path, help="existing independent production metallib for TILED/exact bridge")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if out.exists() or out.is_relative_to(ROOT):
        parser.error("output must be new and outside repository")
    if args.run and not args.deployed_metallib:
        parser.error("--run requires --deployed-metallib")
    shader = (ROOT / "metal/myds_indexer_register_only.metal").read_bytes()
    if digest(shader) != APPROVED:
        raise RuntimeError("approved original shader SHA changed; refusing to derive candidate")
    original = shader.decode()
    candidate = replace_one(original, "args.n_comp > 16384u", "args.n_comp > 65536u")
    candidate = replace_one(candidate, "args.pos0 > 32768u - args.n_tokens", "args.pos0 > 131072u - args.n_tokens")
    fixture = (ROOT / "tools/bench_indexer_register_only.m").read_text()
    case_pattern = r"static const unsigned CASES\[\]\[4\] = \{.*?\n\};"
    cases = """static const unsigned CASES[][4] = {
    {8,16385,32762,0},{9,16385,32761,1},{15,32767,65519,2},
    {31,32768,65505,0},{63,32769,65475,1},{31,65535,131039,2},
    {32,65536,131040,0},{9,65536,131063,1}
};"""
    fixture, count = re.subn(case_pattern, cases, fixture, flags=re.S)
    if count != 1:
        raise RuntimeError("fixture case table anchor changed")
    fixture = replace_one(fixture, "a->n_comp>16384", "a->n_comp>65536")
    fixture = replace_one(fixture, "a->pos0+a->n_tokens>32768", "a->pos0+a->n_tokens>131072")
    fixture = replace_one(fixture, "REJECT(n_comp,16385)", "REJECT(n_comp,65537)")
    fixture = replace_one(fixture, "REJECT(pos0,32761)", "REJECT(pos0,131065)")
    # The historic fixture writes every guarded tensor (hundreds of MiB here).
    # Retain SHA/size/name in the manifest while still comparing in GPU memory.
    start = fixture.index("static NSDictionary *raw(NSString *name,")
    end = fixture.index("static NSDictionary *rawdata(", start)
    fixture = fixture[:start] + '''static NSDictionary *raw(NSString *name,const void *p,size_t bytes) {
    return @{@"name":name,@"bytes":@(bytes),@"sha256":sha(p,bytes),@"stored":@NO};
}
''' + fixture[end:]
    fixture = replace_one(fixture, 'manifest[@"status"]=status;manifest[@"files"]=files;',
                          'manifest[@"status"]=status;manifest[@"files"]=files;manifest[@"buffer_storage"]=@"sha256_only";')
    topk_source = (ROOT / "tools/indexer_long_topk_probe.inc").read_text()
    fixture = replace_one(fixture, "int main(int argc,char **argv){@autoreleasepool {",
                          topk_source + "\nint main(int argc,char **argv){@autoreleasepool {")
    fixture = replace_one(fixture, '    unsigned completed=0;BOOL all=YES;NSString *roles[]=',
        '    if(!host){init_topk(options[@"--deployed-metallib"]);'
        'need([manifest[@"topk_runtime_library"][@"sha256"] isEqual:libraries[0][@"sha256"]],'
        '"topk runtime library matches initial deployed capture");}\n'
        '    unsigned completed=0;BOOL all=YES;NSString *roles[]=')
    fixture = replace_one(fixture, '            }}\n        }\n        [manifest[@"cases"]addObject:c];',
        '''            }}
            NSDictionary *topk=topk_against_exact(logical[4],logical[3],nk,nt,pos);
            c[@"actual_exact_vs_long_top512"]=topk;
            all &= [topk[@"guards_and_causal_ids_valid"]boolValue] && [topk[@"exact_aa_ordered_id_mismatches"]unsignedLongLongValue]==0 && [topk[@"ordered_id_mismatches"]unsignedLongLongValue]==0;
            emit(@{@"kind":@"top512_result",@"case":@(ci),@"details":topk});
        }
        [manifest[@"cases"]addObject:c];''')
    fixture = replace_one(fixture, 'return host||all?0:2;',
                          'if(!host){myds_v41_gpu_scratch_free(topk_scratch);myds_metal_cleanup();}return host||all?0:2;')
    # The frozen builder's metadata otherwise claims its old 32K scope. Its
    # generated copy is isolated and records the actual widened probe bounds.
    builder = (ROOT / "tools/build_indexer_register_only_probe_v2.py").read_text()
    builder = replace_one(builder, "n_comp_inclusive_max=16384,", "n_comp_inclusive_max=65536,")
    builder = replace_one(builder, "end_inclusive_max=32768,", "end_inclusive_max=131072,")
    builder = replace_one(builder,
        '"-framework","Foundation","-framework","Metal",str(source/"tools/bench_indexer_register_only.m"),',
        '"-I",str(root/"include"),"-I",str(root/"src"),"-framework","Foundation","-framework","Metal",'
        'str(source/"tools/bench_indexer_register_only.m"),str(root/"src/myds_v41_gpu.c"),'
        'str(root/"src/myds_metal.m"),str(root/"src/myds_gguf.c"),str(root/"src/myds_log.c"),"-lm","-lpthread",')
    out.mkdir(parents=True)
    (out / "candidate.metal.inc").write_text(candidate)
    (out / "fixture.m").write_text(fixture)
    (out / "isolated_builder.py").write_text(builder)
    runtime_inputs = [ROOT / rel for rel in ("src/myds_v41_gpu.c", "src/myds_metal.m",
                       "src/myds_decode_candidates.inc", "src/myds_gguf.c", "src/myds_log.c")]
    runtime_inputs += sorted((ROOT / "include").glob("*.h"))
    runtime_hashes = {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in runtime_inputs}
    transformation = {"baseline_shader_sha256": digest(shader), "candidate_sha256": digest(candidate.encode()),
        "source_fixture_sha256": digest((ROOT / "tools/bench_indexer_register_only.m").read_bytes()),
        "fixture_sha256": digest(fixture.encode()), "builder_sha256": digest(builder.encode()),
        "changes": ["shader n_comp upper bound 16384 -> 65536", "shader end upper bound 32768 -> 131072",
                    "fixture long cases/scope and SHA-only buffer evidence", "actual runtime Top512 GPU comparison",
                    "builder scope metadata and fixture link inputs"],
        "topk_helper_sha256": digest(topk_source.encode()),
        "generator_sha256": digest(Path(__file__).read_bytes()), "runtime_input_sha256": runtime_hashes,
        "runtime_connected": False, "production_adoption": False}
    (out / "derivation.json").write_text(json.dumps(transformation, indent=2) + "\n")
    command = [sys.executable, str(out / "isolated_builder.py"), "--source-root", str(ROOT),
               "--out-dir", str(out / "build"), "--candidate-source", str(out / "candidate.metal.inc"),
               "--harness-source", str(out / "fixture.m")]
    subprocess.run(command, check=True)
    if any(digest(path.read_bytes()) != runtime_hashes[str(path.relative_to(ROOT))] for path in runtime_inputs):
        raise RuntimeError("runtime source/header changed during compilation")
    exe = out / "build/bench_indexer_register_only"
    subprocess.run([str(exe), "--scope-self-test"], check=True)
    if not args.run:
        print("Build and CPU scope tests passed; no GPU work submitted.")
        return
    require_idle()
    deployed = args.deployed_metallib.resolve(strict=True)
    command = [str(exe), "--native-metallib", str(out / "build/native.metallib"),
               "--probe-metallib", str(out / "build/probe.metallib"), "--deployed-metallib", str(deployed),
               "--build-identity", str(out / "build/source-identity.json"),
               "--harness-source", str(out / "fixture.m"), "--out-dir", str(out / "gpu")]
    (out / "gpu-command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (out / "gpu.stdout.log").open("x") as stream:
        result = subprocess.run(command, stdout=stream)
    # Runtime emits human-readable log lines to stdout too; preserve them but
    # keep the structured event stream parseable, including on a failed gate.
    with (out / "gpu.jsonl").open("x") as structured:
        for line in (out / "gpu.stdout.log").read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            structured.write(json.dumps(row, sort_keys=True) + "\n")
    if result.returncode:
        raise RuntimeError(f"GPU correctness rejected (exit {result.returncode}); evidence: {out / 'gpu/artifact-manifest.json'}")
    print(f"GPU correctness passed; evidence: {out / 'gpu/artifact-manifest.json'}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"long-indexer-probe: {error}", file=sys.stderr)
        sys.exit(1)
