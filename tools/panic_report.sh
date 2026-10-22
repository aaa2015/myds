#!/bin/bash
# =================================================================
# 🔍 panic_report.sh — 读 macOS panic/Jetsam 报告，判断是不是 wired 打满
#
# 为什么需要它：整机 panic 之后，应用日志通常已被轮转或截断，
# 唯一还留着现场的是 /Library/Logs/DiagnosticReports/ 里的 panic 文件。
# 这个脚本把里面真正有用的四个字段挖出来，并对多次 panic 做归因：
#   - 是不是 wired 打满（wired / free / inactive / pagesReclaimed）
#   - 有几个同名大进程（>=2 ⇒ 重复加载同一份模型）
#   - 每个进程的 pageIns / lastRunTime / userTime（区分「还在加载」和「运行中分配」）
#
# 用法：
#   ./panic_report.sh                    # 分析全部 panic，最新的在前
#   ./panic_report.sh --latest           # 只分析最新一份
#   ./panic_report.sh <path.panic>       # 分析指定文件
#   ./panic_report.sh --jetsam           # 附带 Jetsam（OOM）事件摘要
#   ./panic_report.sh --watch-model myds # 只关心某个进程名（默认 myds）
#
# 退出码：0 = 分析完成；1 = 参数错误或没有 panic 文件
# =================================================================

set -u

DIR=/Library/Logs/DiagnosticReports
ONLY_LATEST=0
DO_JETSAM=0
WATCH=myds
TARGET=""

while [ $# -gt 0 ]; do
    case "$1" in
        --latest)      ONLY_LATEST=1; shift ;;
        --jetsam)      DO_JETSAM=1; shift ;;
        --watch-model) WATCH="$2"; shift 2 ;;
        -h|--help)     sed -n '2,22p' "$0"; exit 0 ;;
        -*)            echo "未知参数: $1" >&2; exit 2 ;;
        *)             TARGET="$1"; shift ;;
    esac
done

if [ -n "$TARGET" ]; then
    [ -f "$TARGET" ] || { echo "找不到 $TARGET" >&2; exit 1; }
    FILES="$TARGET"
else
    if [ "$ONLY_LATEST" = "1" ]; then
        F=$(ls -t "$DIR"/panic-full-*.panic 2>/dev/null | head -1)
        [ -n "$F" ] || { echo "没有找到 panic 文件（$DIR）" >&2; exit 1; }
        FILES="$F"
    else
        FILES=$(ls -t "$DIR"/panic-full-*.panic 2>/dev/null)
        [ -n "$FILES" ] || { echo "没有找到 panic 文件（$DIR）" >&2; exit 1; }
    fi
fi

# ---------------------------------------------------------------
# 解析器：一个文件一次调用，输出结构化文本供上层比对
# ---------------------------------------------------------------
parse_panic() {
    /usr/bin/python3 - "$1" "$WATCH" <<'PY'
import sys, re

path, watch = sys.argv[1], sys.argv[2]
raw = open(path, "rb").read().decode("utf-8", "replace")
PAGE = 16384  # Apple Silicon

def gib(pages):
    return pages * PAGE / 2**30

def gb(pages):
    """十进制 GB。与 gib() 并存是刻意的：
    报告里这两个数看起来差 7%，历史上我自己就混用过（说 204.9 GB 其实指 190.8 GiB），
    后来对照才发现。两个都打出来，读者不用换算也不会误会。"""
    return pages * PAGE / 1e9

def both(pages):
    return "%.1f GiB (%.1f GB)" % (gib(pages), gb(pages))

# ── 时间戳 ──
m = re.search(r'"timestamp"\s*:\s*"([^"]+)"', raw)
ts = m.group(1) if m else "?"
print("FILE   %s" % path.split("/")[-1])
print("TIME   %s" % ts)

# ── panicString ──
m = re.search(r'"panicString"\s*:\s*"(.*?)"\s*,\s*\n', raw, re.S)
if m:
    s = m.group(1).replace("\\n", " ").strip()
    print("PANIC  %s" % s[:220])
else:
    print("PANIC  (未找到 panicString)")
    sys.exit(0)

# ── memoryStatus：判断是不是 wired 打满 ──
m = re.search(r'"memoryStatus"\s*:\s*\{', raw)
if m:
    depth, i = 0, m.end() - 1
    while i < len(raw):
        c = raw[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    blk = raw[m.start():i + 1]
    wired = re.search(r'"wired"\s*:\s*(\d+)', blk)
    free = re.search(r'"free"\s*:\s*(\d+)', blk)
    inact = re.search(r'"inactive"\s*:\s*(\d+)', blk)
    reclaimed = re.search(r'"pagesReclaimed"\s*:\s*(\d+)', blk)
    if wired:
        w = int(wired.group(1))
        print("WIRED  %d pages = %s" % (w, both(w)))
        verdict = "WIRED_EXHAUSTED" if w > 10_000_000 else "not-wired-exhaustion"
        if reclaimed and reclaimed.group(1) == "0" and inact and int(inact.group(1)) < 100:
            verdict += " +NOTHING_RECLAIMABLE"
        print("VERDICT %s" % verdict)
    if free:
        print("FREE   %d pages = %.1f MB" % (int(free.group(1)), int(free.group(1)) * PAGE / 2**20))
    if inact:
        print("INACT  %d pages" % int(inact.group(1)))
    if reclaimed:
        print("RECLAIMED %s pages" % reclaimed.group(1))
    if wired:
        w = int(wired.group(1))
        total = re.search(r'"memoryPages"\s*:\s*\{[^}]*\}', blk)
        # 物理内存从 panic 头部取
        tm = re.search(r'"totalMemory(?:Size)?"\s*:\s*(\d+)', raw)
        if tm:
            tot = int(tm.group(1))
            unit = "bytes"
            if tot < 10**6:
                tot *= PAGE; unit = "pages"
            print("RATIO  wired/total = %.1f%%  (%s of %s)"
                  % (100.0 * w * PAGE / tot, both(w), both(tot // PAGE) if unit == "pages" else "%.1f GiB" % (tot / 2**30)))

# ── processByPid：找出同名进程 ──
i = raw.find('"processByPid"')
if i < 0:
    print("PROCS  (无 processByPid)")
    sys.exit(0)

j = raw.find("{", i)
depth, k, ins, esc = 0, j, False, False
while k < len(raw):
    c = raw[k]
    if ins:
        if esc:
            esc = False
        elif c == "\\":
            esc = True
        elif c == '"':
            ins = False
    else:
        if c == '"':
            ins = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
    k += 1
blk = raw[j:k + 1]

hits = []
for m in re.finditer(r'"(\d+)"\s*:\s*', blk):
    start = m.end()
    if blk[start] != "{":
        continue
    d, e, ins, esc = 0, start, False, False
    while e < len(blk):
        c = blk[e]
        if ins:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                ins = False
        else:
            if c == '"':
                ins = True
            elif c == "{":
                d += 1
            elif c == "}":
                d -= 1
                if d == 0:
                    break
        e += 1
    entry = blk[start:e + 1]
    pn = re.search(r'"procname"\s*:\s*"([^"]*)"', entry)
    if not pn or pn.group(1) != watch:
        continue
    rec = {"pid": m.group(1)}
    for kk in ("userTimeTask", "systemTimeTask", "pageIns", "pageFaults",
               "lastRunTime", "jetsamCoalition", "notice"):
        r = re.search(r'"%s"\s*:\s*("([^"]*)"|([\d.]+))' % kk, entry)
        if r:
            rec[kk] = r.group(2) if r.group(2) is not None else r.group(3)
    hits.append(rec)

print("PROCS  %s 进程数=%d" % (watch, len(hits)))
for r in hits:
    pi = r.get("pageIns")
    extra = ""
    if pi and pi.isdigit():
        extra = "  pageIns=%s页=%s" % (pi, both(int(pi)))
#    判断：还在加载 vs 运行中
        if int(pi) > 1_000_000:
            extra += " [还在page-in=启动阶段]"
        elif int(pi) < 100_000:
            extra += " [权重已驻留=运行阶段]"
    print("  pid=%-7s userTime=%-8s sysTime=%-8s lastRun=%-9s jetsam=%s%s"
          % (r.get("pid", "?"), r.get("userTimeTask", "?")[:8],
             r.get("systemTimeTask", "?")[:8], r.get("lastRunTime", "?")[:9],
             r.get("jetsamCoalition", "?"), extra))
    if r.get("notice"):
        print("         notice: %s" % r["notice"][:90])
PY
}

echo "═══════════════════════════════════════════════"
echo " 🔍 Panic Report   watch=$WATCH"
echo "═══════════════════════════════════════════════"

for f in $FILES; do
    parse_panic "$f"
    echo "───────────────────────────────────────────────"
done

# ---------------------------------------------------------------
# 多次 panic 的归因提示
# ---------------------------------------------------------------
N=$(echo "$FILES" | wc -l | tr -d ' ')
if [ "$N" -ge 2 ]; then
    echo "📊 $N 份 panic 报告"
    echo
    echo "  判定同一面墙 / 不同入口的方法："
    echo "    · wired 与 free 几乎相同、但 ${WATCH} 进程数不同  ⇒ 同一失效模式，不同触发"
    echo "    · 进程数 >= 2 且 lastRunTime 差很多            ⇒ 重复加载（旧的没退干净）"
    echo "    · 进程数 = 1 且 pageIns 很小                   ⇒ 权重已驻留，是运行期分配"
    echo
fi

# ---------------------------------------------------------------
# Jetsam（内存压力杀进程，与 panic 不同）
# ---------------------------------------------------------------
if [ "$DO_JETSAM" = "1" ]; then
    echo "═══ JetsamEvent（OOM 杀进程，不是 panic）═══"
    for f in $(ls -t "$DIR"/JetsamEvent-*.ips 2>/dev/null | head -3); do
        /usr/bin/python3 - "$f" <<'PY'
import json, sys
p = sys.argv[1]
lines = open(p).read().split("\n", 1)
try:
    meta = json.loads(lines[0])
    d = json.loads(lines[1])
except Exception as e:
    print("  %s: 解析失败 %s" % (p.split("/")[-1], e))
    raise SystemExit
rows = []
for proc in (d.get("processes") or []):
    rows.append((proc.get("rpages") or 0, proc.get("name", "?")))
rows.sort(reverse=True)
print("  %s  top: %s" % (meta.get("timestamp", "?")[:19],
      ", ".join("%s %.1fGB" % (n, r * 16384 / 2**30) for r, n in rows[:3])))
PY
    done
    echo
    echo "  注意：Jetsam 与 panic 不是一个东西。Jetsam = 内核杀掉某个进程保住系统；"
    echo "        panic = 系统扛不住直接重启。两者可能同源，但处置不同。"
fi

echo "═══════════════════════════════════════════════"
echo " 结论要点："
echo "   · wired 接近物理内存 + pagesReclaimed=0 + watchdog 未签到 ⇒ wired 打满型 panic"
echo "   · 这类 panic 会截断应用日志，现场只能从本文件的事件里找"
echo "   · panic 文件不含 ppid/argv/env ⇒ 谁启动的进程无法从它确定，不要靠时间推测"
echo "═══════════════════════════════════════════════"
