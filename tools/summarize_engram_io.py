#!/usr/bin/env python3
"""Validate actual-token CPU Engram A/B evidence, then summarize timings.

Never equate fresh inode, F_NOCACHE, logical pread bytes or global read growth
with a guarantee that every lookup physically reached NAND. Report each
observation and keep full-model GPU latency outside this scope.
"""
import argparse
import json
import math
from pathlib import Path
import re
import statistics


def hash_u32(values):
    state = 14695981039346656037
    for value in values:
        for byte in (value & 0xFFFFFFFF).to_bytes(4, "little"):
            state = ((state ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return format(state, "016x")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def counters(text):
    out = {}
    for block in text.split("+-o IOBlockStorageDriver")[1:]:
        uid = re.search(r"id (0x[0-9a-f]+)", block)
        items = re.search(r'"Statistics" = \{([^}]+)', block)
        require(uid and items, "unparseable storage driver")
        out[uid.group(1)] = {k: int(v) for k, v in re.findall(r'"([^\"]+)"=(\d+)', items.group(1))}
    require(bool(out), "no IOBlockStorageDriver counters")
    return out


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize(root):
    window = read_jsonl(root / "window.jsonl")
    require(window and window[0]["kind"] == "window_created", "missing window start")
    require(window[-1]["kind"] == "window_complete" and window[-1]["error"] is None and
            window[-1]["source_stat_unchanged"] and window[-1]["clones_removed"], "window incomplete")
    metadata = window[0]
    require(metadata["order"] and all(n in (1, 4, 8) for n in metadata["order"]) and
            1 <= metadata["runs"] <= 8, "invalid measurement order/runs")
    created = [r for r in window if r["kind"] == "clone_created"]
    ended = {r["name"]: r for r in window if r["kind"] == "measurement_end"}
    require(len(created) == len(metadata["order"]), "variant count mismatch")
    # A filesystem can reuse an inode after an earlier clone is deleted. Require
    # each clone to differ from the source, not from a previously deleted clone.
    results = []
    reference_tokens = None
    reference_rows = None
    reference_values = None
    reference_layout = None
    reference_input = None
    reference_value_aggregate = None
    require(len({r["name"] for r in created}) == len(created), "duplicate measurement name")
    for index, clone in enumerate(created):
        name = clone["name"]
        rows = read_jsonl(root / (name + ".jsonl"))
        require(rows and rows[0]["kind"] == "engram_io_ready", name + ": missing ready")
        require(rows[-1] == {"kind": "engram_io_complete", "valid": True}, name + ": missing complete")
        ready = rows[0]
        source = (ready["input_kind"], ready["input_bytes_fnv1a64"], ready["token_ids_fnv1a64"])
        if reference_input is None:
            reference_input = source
        require(source == reference_input, name + ": input metadata differs")
        require(name in ended and ended[name]["returncode"] == 0, name + ": bad exit")
        require(ready["workers"] == clone["workers"] == metadata["order"][index], name + ": wrong workers")
        require(ready["runs"] == metadata["runs"] and ready["mode"] == metadata["mode"], name + ": mode drift")
        require(ready["inode"] == clone["inode"] and ready["device"] == clone["device"] and
                ready["inode"] != metadata["model_stat"]["inode"] and
                ready["file_bytes"] == clone["file_bytes"] == metadata["model_stat"]["bytes"], name + ": clone mismatch")
        layout = (ready["table_offsets"], ready["table_rows"])
        if reference_layout is None:
            reference_layout = layout
        require(layout == reference_layout, name + ": different table layout")
        done = [r for r in rows if r["kind"] == "engram_io_lookup_done"]
        phases = [r for r in rows if r["kind"] == "engram_io_lookup_ready"]
        require([r["run"] for r in done] == list(range(metadata["runs"])) and
                [r["run"] for r in phases] == list(range(metadata["runs"])), name + ": missing run")
        sample = dict(name=name, workers=ready["workers"], token_count=ready["token_count"],
                      input_kind=ready["input_kind"], inode=ready["inode"], runs=[])
        for finish in done:
            run = finish["run"]
            tokens = [r for r in rows if r["kind"] == "engram_io_token" and r["run"] == run]
            require(len(tokens) == finish["token_count"] == ready["token_count"] > 0, name + ": token count")
            require([r["pos"] for r in tokens] == list(range(len(tokens))), name + ": position drift")
            identity = [(r["token"], r["history"]) for r in tokens]
            row_identity = [(r["row_ids"], r["rows_fnv1a64"]) for r in tokens]
            value_identity = [r["values_fnv1a64"] for r in tokens]
            if reference_tokens is None:
                reference_tokens, reference_rows, reference_values = identity, row_identity, value_identity
            require(identity == reference_tokens and row_identity == reference_rows and
                    value_identity == reference_values, name + ": serial/parallel values or input mismatch")
            require(hash_u32([r["token"] for r in tokens]) == ready["token_ids_fnv1a64"],
                    name + ": token aggregate hash mismatch")
            require(all(hash_u32(r["row_ids"]) == r["rows_fnv1a64"] for r in tokens),
                    name + ": row hash mismatch")
            require(hash_u32([rid for r in tokens for rid in r["row_ids"]]) == finish["rows_fnv1a64"],
                    name + ": row aggregate hash mismatch")
            require(all(re.fullmatch(r"[0-9a-f]{16}", r["values_fnv1a64"]) for r in tokens) and
                    re.fullmatch(r"[0-9a-f]{16}", finish["values_fnv1a64"]), name + ": malformed value hash")
            if reference_value_aggregate is None:
                reference_value_aggregate = finish["values_fnv1a64"]
            require(finish["values_fnv1a64"] == reference_value_aggregate, name + ": value aggregate differs")
            require(finish["valid"] and finish["cold_candidate"] == (run == 0), name + ": invalid finish")
            times = [r["prepare_ms"] for r in tokens]
            require(all(math.isfinite(t) and t > 0 for t in times), name + ": bad timings")
            require(math.isclose(sum(times), finish["prepare_ms_sum"], abs_tol=len(times) * 1e-8) and
                    math.isclose(sum(times) / len(times), finish["prepare_ms_per_token"], abs_tol=1e-8), name + ": timing aggregate mismatch")
            require(finish["logical_row_bytes_upper_bound"] == len(tokens) * 48 * 264, name + ": row byte bound")
            require(finish["history"] == tokens[-1]["history"], name + ": final history mismatch")
            before = [r for r in window if r["kind"] == "storage_before" and r["name"] == name and r["run"] == run]
            after = [r for r in window if r["kind"] == "storage_after" and r["name"] == name and r["run"] == run]
            require(len(before) == len(after) == 1, name + ": unbracketed storage counters")
            a, b = counters(before[0]["ioreg"]), counters(after[0]["ioreg"])
            require(a.keys() == b.keys(), name + ": device set changed")
            keys = ["Bytes (Read)", "Operations (Read)", "Total Time (Read)"]
            require(all(all(k in d for k in keys) for d in list(a.values()) + list(b.values())), name + ": incomplete counters")
            delta = {uid: {k: b[uid][k] - a[uid][k] for k in keys} for uid in a}
            require(all(v >= 0 for d in delta.values() for v in d.values()), name + ": counter reset")
            total = {k: sum(d[k] for d in delta.values()) for k in keys}
            blocks = set()
            unique = set()
            for token in tokens:
                require(len(token["row_ids"]) == 48, name + ": incomplete row IDs")
                for j, rid in enumerate(token["row_ids"]):
                    table = j // 24
                    require(0 <= rid < ready["table_rows"][table], name + ": out of range row")
                    unique.add((table, rid))
                    off = ready["table_offsets"][table] + rid * 264
                    blocks.update(range(off // 4096, (off + 263) // 4096 + 1))
            sample["runs"].append(dict(run=run, cold_candidate=(run == 0),
                prepare_ms_per_token=finish["prepare_ms_per_token"],
                prepare_ms_token_median=statistics.median(times),
                distinct_table_rows=len(unique), distinct_4k_reference_blocks=len(blocks),
                reference_4k_bytes=len(blocks) * 4096,
                storage_driver_delta_total=total, storage_driver_delta_by_id=delta,
                device_reads_observed=total["Bytes (Read)"] > 0,
                device_bytes_divided_by_distinct_4k_reference_bytes=total["Bytes (Read)"] / (len(blocks) * 4096),
                logical_row_bytes_upper_bound=finish["logical_row_bytes_upper_bound"],
                rows_fnv1a64=finish["rows_fnv1a64"], values_fnv1a64=finish["values_fnv1a64"]))
        results.append(sample)
    by_workers = {}
    for workers in sorted(set(metadata["order"])):
        selected = [r for r in results if r["workers"] == workers]
        cold = [r["runs"][0]["prepare_ms_per_token"] for r in selected]
        warm = [z["prepare_ms_per_token"] for r in selected for z in r["runs"][1:]]
        by_workers[str(workers)] = dict(fresh_clone_samples=len(cold),
            fresh_clone_prepare_ms_per_token_median=statistics.median(cold),
            fresh_clone_prepare_ms_per_token_values=cold,
            all_fresh_clone_phases_have_device_reads=all(r["runs"][0]["device_reads_observed"] for r in selected),
            warm_prepare_ms_per_token_median=statistics.median(warm) if warm else None,
            warm_prepare_ms_per_token_values=warm)
    if "1" in by_workers:
        base = by_workers["1"]["fresh_clone_prepare_ms_per_token_median"]
        for stats in by_workers.values():
            ms = stats["fresh_clone_prepare_ms_per_token_median"]
            stats.update(fresh_clone_speedup_vs_serial=base / ms,
                         fresh_clone_reduction_percent=100 * (1 - ms / base))
    return dict(scope="Actual-token production CPU prepare-token path; not full-model prefill/decode latency",
        boundary="Each first run uses a fresh clone inode; repeated n-grams can naturally reuse pages within a run. Device read counters exclude GGUF loading and bracket lookup/validation/logging. They are global OS storage-driver counters, not per-process or NAND attribution; fresh inode and F_NOCACHE do not guarantee coldness.",
        identity_validation="all token IDs, histories, row IDs and decoded-value hashes match across every worker and repeat",
        by_workers=by_workers, samples=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = summarize(args.directory)
    out = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    (args.output or args.directory / "summary.json").write_text(out)
    print(json.dumps(result["by_workers"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
