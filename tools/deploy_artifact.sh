#!/bin/zsh
# =================================================================
# 产物部署：macmini 构建 → macstudio 运行，不传源码、不在远端编译
# =================================================================
#
# 背景（2026-10-31 统帅定）：
#   macmini 与 macstudio 环境已对齐，macstudio【不再从源码编译】，只接收产物。
#   此前 sync_myds.sh 的模型相反：它 --exclude='bin' 只传源码，指望远端 make。
#   而 macstudio 上 Metal 工具链不可用（xcrun -f metal 无结果），
#   产物本来就只能在 macmini 上产出 —— 这个方向才是唯一可行的。
#
# 部署的产物（生产进程只打开这两个文件，见 lsof）：
#   bin/myds           可执行文件
#   bin/myds.metallib  Metal 内核库
#
# 重启语义（2026-10-31 改）：fleet.conf 的 8187 已从 E=3 改为 E=1，
#   E=1 不会因产物变化自动重启，所以本脚本【自己负责 stop → start】。
#   停止前会校验 pid 的命令行确实含 bin/myds，避免误杀。
#
# 安全措施：
#   · 传送前把远端旧产物备份到 bin/.prev-<时间戳>/，失败可一条 rsync 回滚
#   · 传送后比对两端 sha256，不一致即 exit 1（不静默）
#   · --dry-run 只显示将做什么
#   · --no-restart 传完不调 start_fleet.sh（默认调用）
#   · 不碰 bin/ 里的其它工具（bench_* / test_* 等），只动这两个生产文件
#
# 用法：
#   ./tools/deploy_artifact.sh                 # 部署并重启
#   ./tools/deploy_artifact.sh --dry-run       # 只看
#   ./tools/deploy_artifact.sh --no-restart    # 只传，不启（下次 start_fleet 生效）

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-macstudio}"
MYDS_SRC="${MYDS_SRC:-$HOME/Projects/myds}"
REMOTE_MYDS="${REMOTE_MYDS:-$HOME/Projects/myds}"
FLEET_DIR="${FLEET_DIR:-$HOME/Projects/myfleet}"

ARTIFACTS=(bin/myds bin/myds.metallib)

DRY_RUN=0
DO_RESTART=1
for arg in "$@"; do
    case "$arg" in
        --dry-run)    DRY_RUN=1 ;;
        --no-restart) DO_RESTART=0 ;;
        -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "未知参数: $arg" >&2; exit 2 ;;
    esac
done

cd "$MYDS_SRC" || { echo "❌ 找不到 $MYDS_SRC" >&2; exit 1; }

# ── 1) 本地产物存在性与身份 ────────────────────────────────────────
echo "📦 本地产物（${MYDS_SRC}）："
for f in "${ARTIFACTS[@]}"; do
    [[ -f "$f" ]] || { echo "❌ 缺少 $f —— 先在 macmini 上 make" >&2; exit 1; }
    printf "   %-20s %8s bytes  %s  %s\n" "$f" \
        "$(stat -f %z "$f")" "$(stat -f '%Sm' -t '%m-%d %H:%M' "$f")" \
        "$(shasum -a 256 "$f" | cut -c1-16)"
done
LOCAL_SRC_HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "   源码 HEAD: ${LOCAL_SRC_HEAD}"

# 干净树检查：脏树构建出的产物无法追到提交，部署它就是不可审计的
if [[ -n "$(git status --porcelain -- src include Makefile 2>/dev/null)" ]]; then
    echo "⚠️  工作树不干净（src/include/Makefile 有未提交改动）"
    echo "   ⇒ 这份产物无法用 HEAD(${LOCAL_SRC_HEAD}) 复现。"
    echo "   继续则部署一个不可审计的二进制。按 Ctrl-C 中止，或 10 秒后继续。"
    [[ $DRY_RUN -eq 1 ]] || sleep 10
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo
    echo "🔍 dry-run：将 rsync ${ARTIFACTS[*]} → ${REMOTE_HOST}:${REMOTE_MYDS}/bin/"
    echo "   （会先备份为 bin/.prev-<时间戳>/）"
    [[ $DO_RESTART -eq 1 ]] && echo "   然后 ssh ${REMOTE_HOST} 'cd ${FLEET_DIR} && ./start_fleet.sh'"
    exit 0
fi

# ── 2) 备份远端旧产物（回滚用）──────────────────────────────────────
STAMP=$(date +%Y%m%d-%H%M%S)
PREV_DIR="bin/.prev-${STAMP}"
echo
echo "🗂  备份远端旧产物 → ${PREV_DIR}"
ssh "$REMOTE_HOST" "cd '$REMOTE_MYDS' && mkdir -p '$PREV_DIR' && \
    for f in ${ARTIFACTS[*]}; do [ -f \"\$f\" ] && cp -p \"\$f\" '$PREV_DIR/' || true; done; \
    ls -1 '$PREV_DIR' | wc -l | tr -d ' '" | \
    xargs -I{} echo "   已备份 {} 个文件"

# ── 3) 传送 ────────────────────────────────────────────────────────
echo "🚚 传送产物 → ${REMOTE_HOST}:${REMOTE_MYDS}/bin/"
for f in "${ARTIFACTS[@]}"; do
    # --no-times：让远端取【当前时间】，否则 E=3 的 mtime 判据可能认为没更新
    rsync -lpgoD --checksum --no-times "$f" "${REMOTE_HOST}:${REMOTE_MYDS}/$f"
done

# ── 4) 自证：两端 sha 一致 ─────────────────────────────────────────
echo "🔎 校验两端 sha256"
MISMATCH=0
for f in "${ARTIFACTS[@]}"; do
    L=$(shasum -a 256 "$f" | awk '{print $1}')
    R=$(ssh "$REMOTE_HOST" "shasum -a 256 '$REMOTE_MYDS/$f' | awk '{print \$1}'")
    if [[ "$L" == "$R" ]]; then
        printf "   ✅ %-20s %s\n" "$f" "${L:0:16}"
    else
        printf "   ❌ %-20s 本地 %s ≠ 远端 %s\n" "$f" "${L:0:16}" "${R:0:16}"
        MISMATCH=1
    fi
done
if [[ $MISMATCH -ne 0 ]]; then
    echo "❌ 产物不一致，不回滚也不重启。人工处理后重跑。" >&2
    exit 1
fi

# ── 5) 重启 ───────────────────────────────────────────────
# 2026-10-31：fleet.conf 里 8187 已从 E=3 改为 E=1。
# E=3 原本靠「产物 mtime 比进程新」自动重启；E=1 只在没跑时启动。
# 所以重启必须由本脚本自己负责 —— 否则产物落地了但进程还跑旧二进制，
# 且 start_fleet.sh 会打印「已经在位」让人以为成功了。
if [[ $DO_RESTART -eq 1 ]]; then
    echo
    echo "🛑 停止 8187（E=1 不会自动重启，由本脚本负责）"
    ssh "$REMOTE_HOST" '
        PID=$(lsof -nP -t -a -iTCP:8187 -sTCP:LISTEN 2>/dev/null | head -1)
        if [ -z "$PID" ]; then echo "   8187 没有在跑，直接启动"; exit 0; fi
        CMD=$(ps -o command= -p "$PID" 2>/dev/null)
        case "$CMD" in
            *bin/myds*) : ;;
            *) echo "   ⚠️ pid $PID 不是 myds，拒绝 kill"; exit 3 ;;
        esac
        echo "   停止 pid $PID"
        kill -TERM "$PID" 2>/dev/null || true
        for i in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 0.2; done
        if kill -0 "$PID" 2>/dev/null; then
            echo "   SIGTERM 无效，SIGKILL"
            kill -9 "$PID" 2>/dev/null || true
            for i in $(seq 1 150); do kill -0 "$PID" 2>/dev/null || break; sleep 0.2; done
        fi
        if kill -0 "$PID" 2>/dev/null; then echo "   ❌ pid $PID 杀不掉"; exit 4; fi
        echo "   已退出，内存已归还"
    '
    STOP_RC=$?
    if [[ $STOP_RC -ne 0 ]]; then
        echo "❌ 停止失败（rc=$STOP_RC），不启动。产物已就位，可人工处理。" >&2
        exit 1
    fi

    echo "🚀 启动"
    ssh "$REMOTE_HOST" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$PATH; \
        cd '$FLEET_DIR' && ./start_fleet.sh" 2>&1 | tail -15
    echo
    echo "🩺 健康检查"
    ssh "$REMOTE_HOST" 'sleep 3; curl -s -m 5 localhost:8187/health || echo "(no response)"'
    echo
else
    echo
    echo "ℹ️  已跳过重启（--no-restart）。产物已就位，但【运行中的进程仍是旧二进制】；"
    echo "   要生效需手动停掉 8187 再跑 start_fleet.sh（E=1 不会因产物变化自动重启）。"
fi

echo
echo "✅ 完成。源码 HEAD=${LOCAL_SRC_HEAD}，备份 ${PREV_DIR}"
echo "   回滚: ssh ${REMOTE_HOST} \"cd ${REMOTE_MYDS} && rsync -a ${PREV_DIR}/ bin/\" 然后 start_fleet.sh"
