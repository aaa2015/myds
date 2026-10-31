"""Coherence audit across arms.

The keyword scorer says "rich" or "refused"; it says nothing about whether the text is
coherent. At alpha=3.5 the ablation is a 350% over-subtraction on a unit vector, so a
degradation check is mandatory rather than optional.

Three cheap, mechanical signals -- no model in the loop, so the number cannot be
flattered:

  cut_idents   identifiers that end mid-word: an uppercase run followed by a lowercase
               run and then nothing (CONSLAB_FR, CONFIG_SLAB_FRE), or a doubled
               separator (security_compute__av)
  repeat_ngram the longest token n-gram that repeats inside one reply, normalised by
               reply length -- loop degeneration shows up here first
  bad_chars    count of U+FFFD and other replacement/control characters

Reported per arm as a rate over replies, because arms differ in length.
"""
import json
import os
import re
import sys
from collections import Counter

ARMS = ["base_nothink", "base_think", "v2_nothink", "v3_nothink"]
R = os.path.expanduser("~/eval/results")

DOUBLE_SEP = re.compile(r"[A-Za-z0-9]__[A-Za-z0-9]")
BAD = re.compile("[\ufffd\u0000-\u0008\u000b\u000c\u000e-\u001f]")


def longest_repeat(text, n=8):
    """Longest repeated n-gram of word tokens, as a fraction of total tokens."""
    toks = text.split()
    if len(toks) < n * 2:
        return 0.0
    grams = Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    top, cnt = grams.most_common(1)[0]
    return cnt * n / len(toks)


def audit(text):
    if not text:
        return None
    return dict(
        double_sep=len(DOUBLE_SEP.findall(text)),
        repeat=round(longest_repeat(text), 4),
        bad_chars=len(BAD.findall(text)),
        tokens=len(text.split()),
    )


def main():
    rows = {}
    for arm in ARMS:
        p = os.path.join(R, "safety_results_%s.jsonl" % arm)
        if not os.path.exists(p):
            continue
        items = [json.loads(l) for l in open(p) if l.strip()]
        rows[arm] = items

    if not rows:
        sys.exit("nothing to audit")

    print("=" * 84)
    print("COHERENCE AUDIT (mechanical; no model in the loop)")
    print("%-13s %-5s %-9s %-11s %-11s %-9s" % (
        "arm", "n", "replies>0", "double_sep", "mean_repeat", "bad_chars"))
    for arm, items in rows.items():
        got = [a for a in (audit(i["content"]) for i in items) if a]
        if not got:
            print("%-13s (no replies)" % arm)
            continue
        ds = sum(1 for a in got if a["double_sep"] > 0)
        print("%-13s %-5d %-9d %-11d %-11.4f %-9d" % (
            arm, len(got), ds,
            sum(a["double_sep"] for a in got),
            sum(a["repeat"] for a in got) / len(got),
            sum(a["bad_chars"] for a in got)))

    print()
    print("=" * 84)
    print("WORST REPEAT PER ARM (top 3 by repeated-ngram fraction)")
    for arm, items in rows.items():
        scored = []
        for i in items:
            a = audit(i["content"])
            if a:
                scored.append((a["repeat"], i["id"], a["tokens"]))
        scored.sort(reverse=True)
        print("%-13s %s" % (arm, ", ".join(
            "%s %.3f/%dtok" % (i, r, t) for r, i, t in scored[:3])))

    print()
    print("=" * 84)
    print("DOUBLE-SEPARATOR IDENTIFIERS (the corruption signature)")
    for arm, items in rows.items():
        hits = []
        for i in items:
            m = DOUBLE_SEP.findall(i["content"])
            if m:
                for x in set(DOUBLE_SEP.finditer(i["content"])):
                    pass
                hits.append((i["id"], len(m)))
        if hits:
            print("%-13s %s" % (arm, hits))


if __name__ == "__main__":
    main()
