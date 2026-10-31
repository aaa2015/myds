"""Add per-request steer_alpha to the eval client so the alpha ladder is runnable.

The server accepts it (myds_server.c:3747), but MydsClient never sent it, so every
ablation arm silently ran at the profile's file-default alpha. That is why the four-arm
grid could only compare glp27 at 3.5 against glp39 at 0.75 -- not a ladder.

Adds:
  MydsClient(steer_alpha=...)   -> payload["steer_alpha"]
  --steer-alpha FLOAT           -> CLI
"""
import shutil
import sys

COMMON = "/Users/fangqingyuan/eval/common.py"

shutil.copy2(COMMON, COMMON + ".bak-alpha")
src = open(COMMON).read()

# 1. constructor parameter
old = """        steer: Optional[str] = None,
        thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ):"""
new = """        steer: Optional[str] = None,
        thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        steer_alpha: Optional[float] = None,
    ):"""
if old not in src:
    sys.exit("could not find the MydsClient signature")
src = src.replace(old, new, 1)

old = """        self.reasoning_effort = str(reasoning_effort) if reasoning_effort is not None else None"""
new = """        self.reasoning_effort = str(reasoning_effort) if reasoning_effort is not None else None
        self.steer_alpha = steer_alpha"""
if old not in src:
    sys.exit("could not find reasoning_effort assignment")
src = src.replace(old, new, 1)

# 2. payload
old = """        if self.steer is not None:
            payload["steer"] = self.steer"""
new = """        if self.steer is not None:
            payload["steer"] = self.steer
        if self.steer_alpha is not None:
            payload["steer_alpha"] = float(self.steer_alpha)"""
if old not in src:
    sys.exit("could not find the steer payload line")
src = src.replace(old, new, 1)

# 3. CLI flag
old = """    parser.add_argument("--steer", choices=["glp27", "glp39", "none"], default=None,"""
new = """    parser.add_argument("--steer-alpha", type=float, default=None,
                        help="Override the steer alpha for this run (server accepts "
                             "steer_alpha per request); without it the profile's "
                             "file-default alpha is used and no ladder is possible")
    parser.add_argument("--steer", choices=["glp27", "glp39", "none"], default=None,"""
if old not in src:
    sys.exit("could not find the --steer argument")
src = src.replace(old, new, 1)

# 4. pass through in resolve_arm_and_client
old = """        reasoning_effort=effort
    )"""
new = """        reasoning_effort=effort,
        steer_alpha=getattr(args, "steer_alpha", None),
    )"""
if old not in src:
    sys.exit("could not find the MydsClient construction")
src = src.replace(old, new, 1)

# 5. echo it in the banner so the log proves which alpha ran
old = """    print(f"[*] Custom Configuration: (steer={steer}, thinking={thinking}, effort={effort})")"""
new = """    print(f"[*] Custom Configuration: (steer={steer}, alpha={getattr(args, 'steer_alpha', None)}, thinking={thinking}, effort={effort})")"""
src = src.replace(old, new, 1)
old = """        print(f"[*] Evaluation Arm: [{arm_info['name']}] (steer={steer}, thinking={thinking}, effort={effort})")"""
new = """        print(f"[*] Evaluation Arm: [{arm_info['name']}] (steer={steer}, alpha={getattr(args, 'steer_alpha', None)}, thinking={thinking}, effort={effort})")"""
src = src.replace(old, new, 1)

open(COMMON, "w").write(src)
print("patched common.py (backup: common.py.bak-alpha)")
