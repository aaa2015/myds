#!/usr/bin/env python3
"""prefill 吞吐claim 的校验器。

存在的唯一理由：2026-11-07 我用生成的英文随机词 prompt 测出 prefill ≈490 t/s，
拿去和台账 §105 的 618.9 t/s 比，得出「掉了 20%」的结论，并据此去二分提交。
用 §105 自己的语料（benchmarks/register_only_2026-10-27/prompts/p32k_ab__v0.txt，
32,224 token）在同一个二进制上重测：619.4 t/s。**差 0.1%。**

真正的变量是 prompt 内容 —— 同 token 数下生成文本比真实语料低 25%，
因为随机词表的 MoE 路由均匀铺满所有专家，没有局部性。

所以这里不检查「数字对不对」（那是生成器的事），只检查一条：
**一个数能不能和台账比，取决于它用的是不是台账那份语料。**
一个声称可比的 claim 必须指到一个在 manifest 里、且 sha256 对得上的语料文件。

另外要求：文档里出现的每个 t/s 字面量都得在寄存器里有一条对应记录，
这样文档和寄存器不可能各说各话。
"""
import hashlib
import json
import re
from pathlib import Path

TOL = 0.05
TPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*t/s")


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(registry, root, corpus=None):
    """返回 (fails, notes)。root 是仓库根；corpus 可注入用于自测。"""
    fails, notes = [], []
    corpus_dir = root / registry["corpus_dir"]
    manifest_path = corpus_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        fails.append(f"manifest 不存在：{manifest_path}")
    ledger_corpus = set(registry.get("ledger_corpus", []))
    claims = registry.get("claims", [])

    # ── 规则 A：可比的 claim 必须指到台账语料，且文件与 manifest 一致 ──────────
    by_file = {}
    for c in claims:
        pf = c.get("prompt_file")
        comparable = bool(c.get("comparable_to_ledger_105"))
        label = f"claim {c.get('value')} t/s（{c.get('where', '?')}）"
        if pf:
            by_file.setdefault(pf, []).append(c)
            path = corpus_dir / pf
            if not path.exists():
                fails.append(f"{label}: 声明的语料文件不存在 {pf}")
                continue
            if pf not in manifest:
                fails.append(f"{label}: {pf} 不在 manifest 里（不是本仓库认定的语料）")
                continue
            got = _sha256(path)
            if got != manifest[pf]["sha256"]:
                fails.append(f"{label}: {pf} 的 sha256 与 manifest 不符 "
                             f"（{got[:16]}… != {manifest[pf]['sha256'][:16]}…）")
            if comparable and pf not in ledger_corpus:
                fails.append(f"{label}: 声称可与台账比，但 {pf} 不在 ledger_corpus 里")
        else:
            if comparable:
                fails.append(f"{label}: 声称可与台账 §105 比，却没有指任何语料文件 —— "
                             f"这正是 2026-11-07 那次误判的形状")
            if c.get("kind") == "prefill" and not c.get("source"):
                fails.append(f"{label}: 没有 prompt 文件，就必须写明 source（为什么不可比）")

    # ── 规则 B：文档里的每个 t/s 字面量都要有登记 ────────────────────────────
    values = [float(c["value"]) for c in claims]
    for doc in registry.get("docs", []):
        p = root / doc
        if not p.exists():
            fails.append(f"登记的文档不存在：{doc}")
            continue
        text = p.read_text(encoding="utf-8")
        seen = {}
        for m in TPS_RE.finditer(text):
            v = float(m.group(1))
            seen.setdefault(round(v, 2), m.start())
        for v, pos in sorted(seen.items()):
            if not any(abs(v - rv) <= TOL for rv in values):
                line = text[:pos].count("\n") + 1
                fails.append(f"{doc}:{line} 出现 {v} t/s 但寄存器里没有对应记录")

    notes.append(f"{len(claims)} 条 claim，{len(by_file)} 个语料文件，"
                 f"{len(registry.get('docs', []))} 篇文档")
    return fails, notes


def self_test(root):
    """证明这套检查有牙：两种坏 claim 必须被拒。"""
    corpus_dir = root / "benchmarks" / "register_only_2026-10-27" / "prompts"
    man = json.loads((corpus_dir / "manifest.json").read_text())
    good_file = sorted(man)[0]
    base = {
        "corpus_dir": "benchmarks/register_only_2026-10-27/prompts",
        "ledger_corpus": [good_file],
        "docs": [],
        "claims": [],
    }

    def run(claims):
        reg = dict(base, claims=claims)
        return validate(reg, root)[0]

    cases = []
    cases.append((
        "无可比、却声称可比（2026-11-07 那次误判的形状）",
        [{"value": 490.0, "kind": "prefill", "prompt_file": None,
          "source": "synthetic", "comparable_to_ledger_105": True}],
        True,
    ))
    cases.append((
        "可比、指到真语料",
        [{"value": 619.4, "kind": "prefill", "prompt_file": good_file,
          "source": "corpus", "comparable_to_ledger_105": True}],
        False,
    ))
    fails = []
    for name, claims, want_fail in cases:
        ok = (bool(run(claims)) == want_fail)
        print(f"  [{'PASS' if ok else 'FAIL'}] 自测：{name}")
        if not ok:
            fails.append(name)

    # 第三种：文件是真的语料，但不在 ledger_corpus 名单里 —— 也不算可比。
    reg = dict(base, ledger_corpus=[], claims=[{
        "value": 619.4, "kind": "prefill", "prompt_file": good_file,
        "source": "corpus", "comparable_to_ledger_105": True}])
    name = "可比、但文件不在 ledger_corpus"
    ok = bool(validate(reg, root)[0])
    print(f"  [{'PASS' if ok else 'FAIL'}] 自测：{name}")
    if not ok:
        fails.append(name)
    return fails


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parent.parent
    reg = json.loads((root / "benchmarks" / "prefill_claims.json").read_text())
    f, n = validate(reg, root)
    for note in n:
        print(note)
    for x in f:
        print("FAIL", x)
    sys.exit(1 if f else 0)
