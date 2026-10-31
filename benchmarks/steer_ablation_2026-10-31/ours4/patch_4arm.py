"""Add the v3_nothink arm + a warmup discard, so the 4-arm grid is
base_nothink / base_think / v2_nothink / v3_nothink.

Two independent fixes:

1. v3_nothink: the previous grid had v3 only in thinking mode. Thinking arms are
   unusable for scoring (the refusal lands in the reasoning channel, §134), so the
   third ablation arm has to exist in non-thinking mode to be comparable.

2. warmup: the first request after a steer-profile switch is not reproducible
   (§126/§128 -- same prompt, three different answers on the first shot, then stable).
   Fire one throwaway request per arm before the scored loop.
"""
import re
import shutil
import sys

COMMON = "/Users/fangqingyuan/eval/common.py"
BENCH = "/Users/fangqingyuan/eval/bench_safety.py"

V3_NOTHINK = '''    "v3_nothink": {
        "name": "v3非思考 (v3-GLP27 NoThink)",
        "desc": "Spec-v3 双挂钩消融 (alpha=3.5, L11-37), 非思考模式",
        "steer": "glp27",
        "thinking": False,
        "effort": "none",
        "tag": "v3_nothink",
    },
'''

WARMUP_FN = '''

WARMUP_PROMPT = "Reply with the single word: ready"


def warmup(client, prompt: str = WARMUP_PROMPT):
    """Burn one request so the scored calls do not include the first shot.

    The first request after a steer-profile switch is not reproducible: same prompt,
    same settings, three different answers, then stable. That first shot is a warmup
    artifact, not a sample, so it must not be scored.
    """
    try:
        kw = {}
        if getattr(client, "thinking", False):
            kw["max_tokens"] = 512
        else:
            kw["max_tokens"] = 32
        res = client.chat(prompt=prompt, **kw)
        print("[*] warmup discarded: %d tok, %.1fs, content=%r" % (
            res["tokens"], res["duration_s"], res["content"][:40]))
        return res
    except Exception as exc:
        print("[!] warmup failed (continuing): %s" % exc)
        return None
'''


def main():
    for path in (COMMON, BENCH):
        shutil.copy2(path, path + ".bak-4arm")
    print("backed up -> *.bak-4arm")

    src = open(COMMON).read()

    # ---- 1. add the v3_nothink arm right after v3_think ----
    if '"v3_nothink"' in src:
        print("common.py: v3_nothink already present")
    else:
        anchor = '        "tag": "v3_think",\n    },\n'
        if anchor not in src:
            sys.exit("common.py: could not find the v3_think arm to anchor on")
        src = src.replace(anchor, anchor + V3_NOTHINK, 1)
        print("common.py: added v3_nothink arm")

    # ---- 2. warmup helper ----
    if "def warmup(" in src:
        print("common.py: warmup already present")
    else:
        anchor = "class MydsClient:"
        if anchor not in src:
            sys.exit("common.py: could not find class MydsClient")
        src = src.replace(anchor, WARMUP_FN.strip("\n") + "\n\n\n" + anchor, 1)
        print("common.py: added warmup()")

    # ---- 3. --no-warmup flag ----
    if "--no-warmup" in src:
        print("common.py: --no-warmup already present")
    else:
        anchor = '    parser.add_argument("--steer",'
        if anchor not in src:
            sys.exit("common.py: could not find --steer to anchor the flag")
        flag = ('    parser.add_argument("--no-warmup", dest="warmup", '
                'action="store_false", default=True,\n'
                '                        help="Skip the discarded warmup request '
                'that normally precedes each arm")\n')
        src = src.replace(anchor, flag + anchor, 1)
        print("common.py: added --no-warmup")

    # ---- 4. arm help text ----
    src = src.replace(
        'help="Evaluation arm (base_nothink, base_think, v2_nothink, v3_think)"',
        'help="Evaluation arm (base_nothink, base_think, v2_nothink, '
        'v3_nothink, v3_think)"')

    open(COMMON, "w").write(src)

    # ---- 5. call warmup in bench_safety before the scored loop ----
    b = open(BENCH).read()
    if "warmup(" in b:
        print("bench_safety.py: warmup call already present")
    else:
        b = b.replace(
            "    load_existing_results,\n    append_result,\n)",
            "    load_existing_results,\n    append_result,\n    warmup,\n)", 1)
        anchor = "    pbar = tqdm(problems"
        if anchor not in b:
            sys.exit("bench_safety.py: could not find the tqdm loop to anchor on")
        call = ('    # first shot after a profile switch is not a sample; discard it\n'
                '    if getattr(args, "warmup", True) and problems:\n'
                '        warmup(client)\n\n')
        b = b.replace(anchor, call + anchor, 1)
        print("bench_safety.py: added warmup call")
        open(BENCH, "w").write(b)

    print("done")


if __name__ == "__main__":
    main()
