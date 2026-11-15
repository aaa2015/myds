#!/usr/bin/env python3
"""tools/evo_memory.py — Structured Evolution Memory & Dead-End Index for myds.

Bridges the 21,000+ line human ledger (`docs/2026-10-20-prefill-decode-pipeline-
and-optimization-ledger.md`) and autonomous trials (`benchmarks/evo_trials.jsonl`)
into a fast, queryable JSONL index (`docs/ledger-index.jsonl`).

Design principle (same as `tools/gen_ledger_tables.py`):
  `docs/ledger-index.jsonl` is 100% deterministically generated from the ledger
  plus `benchmarks/evo_trials.jsonl`. `--check` verifies byte-exact freshness so
  the index never drifts from the authoritative ledger.

Modes:
  python3 tools/evo_memory.py --write
  python3 tools/evo_memory.py --check
  python3 tools/evo_memory.py --query --subsystem indexer [--verdict REJECT] [--json]
  python3 tools/evo_memory.py --query --kernel topk [--limit 20]
  python3 tools/evo_memory.py --record --subsystem indexer --hypothesis "..." \\
      --verdict REJECT --why "..." [--delta-pct -1.2] [--ulp-zero true]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = ROOT / "docs" / "2026-10-20-prefill-decode-pipeline-and-optimization-ledger.md"
INDEX_PATH = ROOT / "docs" / "ledger-index.jsonl"
TRIALS_PATH = ROOT / "benchmarks" / "evo_trials.jsonl"

KNOB_RE = re.compile(r"\bMYDS_[A-Z0-9_]+\b")
KERNEL_RE = re.compile(r"\bkernel_[a-zA-Z0-9_]+\b")
SEC_H2_RE = re.compile(r"^##\s+(?:§\s*)?(\d+)(?:[.:：\s]+)(.+)$")
DATE_RE = re.compile(r"(2026-\d{2}-\d{2})")

SUBSYSTEM_PATTERNS: list[tuple[str, str]] = [
    ("indexer", r"indexer|top-?k|top-?512|top-?2048|argsort|radix|register[-_ ]only|\biro\b|yoco"),
    ("moe", r"\bmoe\b|iq2_xxs|q2_k|expert|router|route|nsg|down_nsg|gateup|decode_topk"),
    ("attn", r"attn|attention|flash_attn|mla\b|rope|sdpa|heads_reg_o"),
    ("engram", r"engram|pread|f_nocache"),
    ("kv_cache", r"kvdsk|kvdisk|prefix[ _-]?cache|compressed|e4m3|macroblock|chunk_size"),
    ("hc_residual", r"\bhc[ _]|hyper-?connect|rmsnorm|norm_mix"),
    ("dense_gemv", r"q4_k|gemv|lm_head|shared[ _]expert|f16_gemv"),
    ("dspark", r"dspark|dspine|speculative|\bpld\b|drafter|depth2"),
    ("vision", r"vision|vispre|visw|visfwd|vit\b"),
    ("tokenizer", r"tokenizer|bpe\b|分词"),
    ("server_window", r"8187|prod_guard|service_window|sigsegv|panic|teardown|/trace|strict_config"),
    ("measurement", r"kprof|0-ulp|口径|测量|假阳性|gpu_gap|剂量"),
]


def detect_subsystems(text: str) -> list[str]:
    low = text.lower()
    found = [name for name, pat in SUBSYSTEM_PATTERNS if re.search(pat, low)]
    return found or ["general"]


def strip_md(s: str) -> str:
    s = re.sub(r"~~([^~]+)~~", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_md_table(lines: list[str], start_line_no: int) -> list[tuple[int, list[str]]]:
    rows: list[tuple[int, list[str]]] = []
    header_seen = False
    for offset, raw in enumerate(lines):
        ln = raw.strip()
        if not ln.startswith("|"):
            if header_seen and rows:
                break
            header_seen = False
            continue
        inner = ln.strip("|")
        if set(inner.replace("|", "").strip()) <= set("-: "):
            header_seen = True
            continue
        cells = [c.strip() for c in inner.split("|")]
        if not header_seen:
            header_seen = True
            continue
        rows.append((start_line_no + offset, cells))
    return rows


def extract_adoption_register(lines: list[str]) -> list[dict[str, Any]]:
    start = end = None
    for i, ln in enumerate(lines):
        if ln.startswith("### 0.3 "):
            start = i
        elif start is not None and (ln.startswith("#### 0.3.1") or ln.startswith("### 0.4")):
            end = i
            break
    if start is None:
        return []
    end = end or len(lines)
    records = []
    status_map = {
        "在产": "ADOPT",
        "已取代": "SUPERSEDED",
        "已丢失": "LOST",
        "已证否": "REJECT",
    }
    for lineno, cells in parse_md_table(lines[start:end], start + 1):
        if len(cells) < 5:
            continue
        ref, content, gain, ulp_col, status_col = cells[:5]
        joined = " ".join(cells)
        status = next((k for k in status_map if k in status_col), "UNKNOWN")
        verdict = status_map.get(status, "UNKNOWN")
        ulp_zero = ("是" in ulp_col) and ("非 0-ULP" not in ulp_col) and not ulp_col.startswith("**否**")
        records.append(
            {
                "kind": "adoption_register",
                "id": f"reg:L{lineno}",
                "source_line": lineno,
                "ref": strip_md(ref),
                "status": status,
                "verdict": verdict,
                "ulp_zero": ulp_zero,
                "subsystems": detect_subsystems(joined),
                "knobs": sorted(set(KNOB_RE.findall(joined))),
                "kernels": sorted(set(KERNEL_RE.findall(joined))),
                "gain": strip_md(gain),
                "summary": strip_md(content)[:320],
            }
        )
    return records


def extract_disproven_rules(lines: list[str]) -> list[dict[str, Any]]:
    start = end = None
    for i, ln in enumerate(lines):
        if ln.startswith("### 0.4 "):
            start = i
        elif start is not None and ln.startswith("### 0.7 "):
            end = i
            break
    if start is None:
        return []
    end = end or len(lines)

    records: list[dict[str, Any]] = []
    cur_h = "0.4"
    i = start
    while i < end:
        ln = lines[i]
        if ln.startswith("### ") or ln.startswith("#### "):
            cur_h = strip_md(ln.lstrip("#").strip())
            i += 1
            continue
        if ln.strip().startswith("|"):
            block_end = i
            while block_end < end and lines[block_end].strip().startswith("|"):
                block_end += 1
            is_conditional = "外部条件变化" in cur_h
            is_fallacy = "0.5.1" in cur_h or "结构特征" in cur_h
            for lineno, cells in parse_md_table(lines[i:block_end], i + 1):
                if len(cells) < 2:
                    continue
                joined = " | ".join(strip_md(c) for c in cells)
                records.append(
                    {
                        "kind": "disproven_rule",
                        "id": f"rule:L{lineno}",
                        "source_line": lineno,
                        "group": cur_h[:80],
                        "verdict": "CONDITIONAL_UNLOCK" if is_conditional else "REJECT",
                        "rule_type": (
                            "conditional_unlock"
                            if is_conditional
                            else ("methodology_trap" if is_fallacy else "disproven_path")
                        ),
                        "subsystems": detect_subsystems(joined + " " + cur_h),
                        "knobs": sorted(set(KNOB_RE.findall(joined))),
                        "kernels": sorted(set(KERNEL_RE.findall(joined))),
                        "summary": joined[:340],
                    }
                )
            i = block_end
            continue
        i += 1
    return records


def classify_section_verdict(title: str, body_head: str) -> str:
    t = title
    if any(w in t for w in ("证否", "不采纳", "驳回", "无效", "排除", "失败", "不可优化")):
        if any(w in t for w in ("采纳", "上线", "通过")) and "不采纳" not in t:
            return "MIXED"
        return "REJECT"
    if any(w in t for w in ("采纳", "投产", "联合上线", "落锤", "已采纳")):
        return "ADOPT"
    if any(w in t for w in ("修掉", "修复", "清理", "收敛", "配对检查", "崩溃", "panic")):
        return "INFRA_FIX"
    if any(w in t for w in ("测量纪律", "口径", "普查", "分解", "归因", "复核", "总账", "时间线")):
        return "ANALYSIS_OR_AUDIT"
    bh = body_head[:600]
    if "不采纳" in bh or "已证否" in bh:
        return "REJECT"
    if "已采纳" in bh or "写入 `myfleet/myds.conf`" in bh:
        return "ADOPT"
    return "ANALYSIS_OR_AUDIT"


def summarize_section_body(body_lines: list[str]) -> str:
    picked: list[str] = []
    priority_re = re.compile(r"结论|结果|判据|收益|根因|原因|采纳|证否|驳回|总结")
    for raw in body_lines[:80]:
        s = raw.strip()
        if not s or s.startswith("```") or s.startswith("|---") or s.startswith("#"):
            continue
        if priority_re.search(s):
            picked.append(strip_md(s.lstrip(">*- ")))
            if len(" ".join(picked)) >= 260:
                break
    if not picked:
        for raw in body_lines[:25]:
            s = raw.strip()
            if s and not s.startswith(("#", "```", "|---")):
                picked.append(strip_md(s.lstrip(">*- ")))
                if len(" ".join(picked)) >= 220:
                    break
    return " / ".join(picked)[:340]


def extract_sections(lines: list[str]) -> list[dict[str, Any]]:
    sec_starts: list[tuple[int, int, str]] = []
    for i, ln in enumerate(lines):
        m = SEC_H2_RE.match(ln)
        if m:
            sec_starts.append((i, int(m.group(1)), m.group(2).strip()))

    records: list[dict[str, Any]] = []
    for idx, (start_i, sec_num, raw_title) in enumerate(sec_starts):
        end_i = sec_starts[idx + 1][0] if idx + 1 < len(sec_starts) else len(lines)
        body_lines = lines[start_i + 1 : end_i]
        body_text = "\n".join(body_lines)
        clean_title = strip_md(raw_title)
        dm = DATE_RE.search(raw_title) or DATE_RE.search(body_text[:400])
        verdict = classify_section_verdict(clean_title, body_text[:1000])
        records.append(
            {
                "kind": "ledger_section",
                "id": f"§{sec_num}",
                "section_num": sec_num,
                "source_line": start_i + 1,
                "date": dm.group(1) if dm else None,
                "title": clean_title[:200],
                "verdict": verdict,
                "subsystems": detect_subsystems(clean_title + "\n" + body_text[:2500]),
                "knobs": sorted(set(KNOB_RE.findall(clean_title + "\n" + body_text))),
                "kernels": sorted(set(KERNEL_RE.findall(clean_title + "\n" + body_text))),
                "summary": summarize_section_body(body_lines),
            }
        )
    return records


def load_autonomous_trials() -> list[dict[str, Any]]:
    if not TRIALS_PATH.exists():
        return []
    rows = []
    for lineno, raw in enumerate(TRIALS_PATH.read_text().splitlines(), 1):
        s = raw.strip()
        if not s:
            continue
        obj = json.loads(s)
        obj.setdefault("kind", "autonomous_trial")
        obj.setdefault("source_line", lineno)
        rows.append(obj)
    return rows


def build_index_text() -> str:
    if not LEDGER_PATH.exists():
        raise RuntimeError(f"Ledger not found: {LEDGER_PATH}")
    lines = LEDGER_PATH.read_text(errors="replace").splitlines()
    all_rows: list[dict[str, Any]] = []
    all_rows.extend(extract_adoption_register(lines))
    all_rows.extend(extract_disproven_rules(lines))
    all_rows.extend(extract_sections(lines))
    all_rows.extend(load_autonomous_trials())
    return "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for r in all_rows
    )


def load_index_rows() -> list[dict[str, Any]]:
    if INDEX_PATH.exists():
        return [
            json.loads(ln)
            for ln in INDEX_PATH.read_text().splitlines()
            if ln.strip()
        ]
    return [json.loads(ln) for ln in build_index_text().splitlines() if ln.strip()]


def query_index(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_index_rows()
    matched = []
    for r in rows:
        if args.subsystem and args.subsystem.lower() not in [
            x.lower() for x in r.get("subsystems", [])
        ]:
            continue
        if args.verdict and r.get("verdict", "").upper() != args.verdict.upper():
            continue
        if args.knob:
            if not any(args.knob.lower() in k.lower() for k in r.get("knobs", [])):
                continue
        if args.kernel:
            hay = " ".join(r.get("kernels", [])) + " " + r.get("title", "") + " " + r.get("summary", "")
            if args.kernel.lower() not in hay.lower():
                continue
        if args.text:
            blob = json.dumps(r, ensure_ascii=False).lower()
            if args.text.lower() not in blob:
                continue
        matched.append(r)

    disproven = [r for r in matched if r["kind"] == "disproven_rule"]
    adopted = [r for r in matched if r["kind"] == "adoption_register"]
    trials = [r for r in matched if r["kind"] == "autonomous_trial"]
    sections = sorted(
        [r for r in matched if r["kind"] == "ledger_section"],
        key=lambda x: -int(x.get("section_num", 0)),
    )
    return {
        "query": {
            "subsystem": args.subsystem,
            "verdict": args.verdict,
            "knob": args.knob,
            "kernel": args.kernel,
            "text": args.text,
        },
        "counts": {
            "total_matched": len(matched),
            "disproven_rules": len(disproven),
            "adoption_register": len(adopted),
            "autonomous_trials": len(trials),
            "ledger_sections": len(sections),
        },
        "disproven_rules": disproven[: args.limit],
        "adoption_register": adopted[: args.limit],
        "autonomous_trials": list(reversed(trials))[: args.limit],
        "ledger_sections": sections[: args.limit],
    }


def record_trial(args: argparse.Namespace) -> dict[str, Any]:
    if not args.hypothesis or not args.verdict or not args.why:
        raise SystemExit("--record requires --hypothesis, --verdict, and --why")
    verdict = args.verdict.upper()
    if verdict not in {"ADOPT", "REJECT", "VOID_MEASUREMENT", "INVESTIGATED_SKIP"}:
        raise SystemExit(
            f"Invalid --verdict '{verdict}' (expected ADOPT|REJECT|VOID_MEASUREMENT|INVESTIGATED_SKIP)"
        )
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    digest = hashlib.sha256(f"{ts}:{args.hypothesis}:{args.why}".encode()).hexdigest()[:10]
    subsystems = (
        [s.strip() for s in args.subsystem.split(",") if s.strip()]
        if args.subsystem
        else detect_subsystems(args.hypothesis + " " + args.why)
    )
    entry: dict[str, Any] = {
        "kind": "autonomous_trial",
        "id": f"trial:{digest}",
        "timestamp_utc": ts,
        "hardware": args.hardware or "unspecified",
        "subsystems": subsystems,
        "knobs": [k.strip() for k in (args.knob or "").split(",") if k.strip()],
        "kernels": [k.strip() for k in (args.kernel or "").split(",") if k.strip()],
        "hypothesis": args.hypothesis.strip(),
        "verdict": verdict,
        "delta_pct": args.delta_pct,
        "ulp_zero": None
        if args.ulp_zero is None
        else (args.ulp_zero.lower() in {"1", "true", "yes"}),
        "summary": args.why.strip(),
    }
    TRIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRIALS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    INDEX_PATH.write_text(build_index_text(), encoding="utf-8")
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Regenerate docs/ledger-index.jsonl")
    mode.add_argument("--check", action="store_true", help="Verify docs/ledger-index.jsonl is fresh")
    mode.add_argument("--query", action="store_true", help="Query structured evolution memory")
    mode.add_argument("--record", action="store_true", help="Record an autonomous trial outcome")

    ap.add_argument("--subsystem", help="Subsystem filter (indexer, moe, attn, engram, kv_cache, ...)")
    ap.add_argument("--verdict", help="Verdict filter (ADOPT, REJECT, MIXED, ...)")
    ap.add_argument("--knob", help="MYDS_* knob substring or comma list")
    ap.add_argument("--kernel", help="Kernel name substring or comma list")
    ap.add_argument("--text", help="Free-text substring filter")
    ap.add_argument("--limit", type=int, default=25, help="Max entries per bucket in --query")
    ap.add_argument("--json", action="store_true", help="Emit JSON output")

    # --record options
    ap.add_argument("--hypothesis", help="What candidate change or mechanism was tested")
    ap.add_argument("--why", help="Measured result & physical reason for ADOPT/REJECT")
    ap.add_argument("--delta-pct", type=float, help="Measured throughput/latency delta %")
    ap.add_argument("--ulp-zero", help="Whether candidate preserved 0-ULP (true/false)")
    ap.add_argument("--hardware", help="Hardware identifier (e.g. 'Apple M2 Ultra 192GB')")

    args = ap.parse_args()

    if args.write:
        txt = build_index_text()
        INDEX_PATH.write_text(txt, encoding="utf-8")
        n_lines = txt.count("\n")
        print(f"Wrote {INDEX_PATH} ({n_lines} structured records)")
        return 0

    if args.check:
        want = build_index_text()
        if not INDEX_PATH.exists():
            print(f"STALE: {INDEX_PATH} missing (run `python3 tools/evo_memory.py --write`)")
            return 1
        got = INDEX_PATH.read_text(encoding="utf-8")
        if got != want:
            print(f"STALE: {INDEX_PATH} differs from ledger (run `python3 tools/evo_memory.py --write`)")
            return 1
        print(f"PASS: {INDEX_PATH.name} is fresh ({want.count(chr(10))} records)")
        return 0

    if args.record:
        entry = record_trial(args)
        if args.json:
            print(json.dumps(entry, indent=2, ensure_ascii=False))
        else:
            print(f"Recorded {entry['id']} [{entry['verdict']}] -> refreshed {INDEX_PATH.name}")
        return 0

    res = query_index(args)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    c = res["counts"]
    print(
        f"== Evolution Memory Query: total={c['total_matched']} "
        f"(disproven_rules={c['disproven_rules']}, adopted={c['adoption_register']}, "
        f"trials={c['autonomous_trials']}, sections={c['ledger_sections']})"
    )
    if res["disproven_rules"]:
        print("\n-- Disproven Guardrails (§0.4 / §0.5.1 / §0.6: DO NOT REPEAT) --")
        for r in res["disproven_rules"]:
            print(f"  [{r['verdict']}] L{r['source_line']} ({r['group']}): {r['summary']}")
    if res["adoption_register"]:
        print("\n-- Adoption Register (§0.3) --")
        for r in res["adoption_register"]:
            print(f"  [{r['status']}] {r['ref']}: {r['gain']} | {r['summary'][:140]}")
    if res["autonomous_trials"]:
        print("\n-- Autonomous Trials --")
        for r in res["autonomous_trials"]:
            print(f"  [{r['verdict']}] {r['id']} ({r.get('hardware')}): {r['hypothesis']} -> {r['summary']}")
    if res["ledger_sections"]:
        print("\n-- Historical Ledger Sections (newest first) --")
        for r in res["ledger_sections"]:
            print(f"  {r['id']} [{r['verdict']}] ({r.get('date') or '-'}) {r['title']}")
            if r["summary"]:
                print(f"      -> {r['summary'][:180]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
