#!/bin/bash
# with_prod_guard.sh — 万能包装器：跑任何会动生产 8187 的命令，结束后自动恢复生产。
#
# 起因（账本 §164）：2026-11-08 07:10–07:25，生产 8187 被反复起停 4 次，
# 最后一次停止后没有任何脚本把它拉回来，停摆 6 分 32 秒。
# 仓库里有 48 个脚本引用 8187、六个 shell 验收脚本各自 kill 生产，
# 其中两个 Python 基准还用 `pgrep -x myds` 杀掉【所有】myds 进程。
# 逐个改不现实，所以提供一个套在任何命令外面的守卫。
#
# 用法：
#   tools/with_prod_guard.sh python3 tools/bench_register_only_ab.py --rounds 2
#   tools/with_prod_guard.sh ./tools/accept_rows4_ulp.sh
#   tools/with_prod_guard.sh --dryrun bash -c 'kill -TERM $PROD_PID'   # 只预演
#
# 行为：
#   1. 记录当前 8187 上的生产进程（完整命令行 + cwd），规范化成绝对路径；
#   2. 原样执行传入的命令（stdin/stdout/exit code 全部透传）；
#   3. 命令以任何方式结束（正常 / 非零 / Ctrl-C / SIGTERM / SIGHUP）后，
#      若 8187 无人监听，则按第 1 步记下的命令行重新拉起生产；
#   4. 若 8187 已被别人占用，什么都不做。
#
# 环境：
#   MYDS_GUARD_PORT    默认 8187
#   MYDS_GUARD_LOG     重拉时的追加日志，默认 myfleet/logs/local-deepseek-flash.log
#   MYDS_GUARD_DRYRUN  非空 ⇒ 只打印将执行的恢复命令
#
# 退出码 = 被包装命令的退出码（恢复失败不影响它，但会打到 stderr）。

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT="${MYDS_GUARD_PORT:-8187}"
LOG="${MYDS_GUARD_LOG:-$HOME/Projects/myfleet/logs/local-deepseek-flash.log}"

DRYRUN="${MYDS_GUARD_DRYRUN:-}"
if [ "${1:-}" = "--dryrun" ]; then DRYRUN=1; shift; fi

if [ $# -eq 0 ]; then
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2
fi

owner() { lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1; }

RESTORE=""
capture() {
    local pid raw cmd argv0 rest cwd abs0
    pid="$(owner)"
    [ -z "$pid" ] && { echo "[prod_guard] $PORT 上没有生产实例（脚本可能已经把它停了）"; return 0; }
    raw="$(ps -o command= -p "$pid" 2>/dev/null)"
    cmd="${raw#"${raw%%[![:space:]]*}"}"
    [ -z "$cmd" ] && return 0
    cwd="$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1)"
    [ -z "$cwd" ] && cwd="$HOME/Projects/myfleet"
    argv0="${cmd%% *}"
    rest="${cmd#"$argv0"}"
    case "$argv0" in /*) abs0="$argv0" ;; *) abs0="$cwd/$argv0" ;; esac
    abs0="$(cd "$(dirname "$abs0")" 2>/dev/null && printf '%s/%s' "$(pwd -P)" "$(basename "$abs0")")"
    [ -z "$abs0" ] && abs0="$argv0"
    RESTORE="cd '$cwd' && exec $abs0$rest"
    echo "[prod_guard] 记录生产 pid=$pid"
    echo "[prod_guard]   恢复命令: $RESTORE"
}

restore() {
    [ -z "$RESTORE" ] && return 0
    if [ -n "$(owner)" ]; then
        echo "[prod_guard] $PORT 已被占用，跳过恢复"
        return 0
    fi
    if [ -n "$DRYRUN" ]; then
        echo "[prod_guard][dryrun] 将执行: $RESTORE"
        return 0
    fi
    echo "[prod_guard] 恢复生产 …"
    nohup sh -c "$RESTORE" >>"$LOG" 2>&1 &
    echo "[prod_guard] 已拉起 pid=$!"
}

trap 'restore' EXIT
trap 'restore; exit 130' INT
trap 'restore; exit 143' TERM HUP

capture
echo "[prod_guard] 执行: $*"
"$@"
rc=$?
echo "[prod_guard] 被包装命令退出码 = $rc"
exit "$rc"
