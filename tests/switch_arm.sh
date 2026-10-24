#!/bin/bash
#
# 部署目录由 MYDS_FLEET_DIR 指定（本仓库不含该目录）。
# 未设置时这些脚本会明确报错退出，而不是猜一个路径。

set -euo pipefail

NR0="${1:-2}"
MASK="${2:-0x04}"

echo "=== Switching Fleet to NR0=${NR0}, MASK=${MASK} ==="
export MYDS_DENSE_Q4_NR0="${NR0}"
export MYDS_DENSE_Q4_NR4_MASK="${MASK}"

OLD_PID=$(lsof -ti :8187 || true)
if [ -n "$OLD_PID" ]; then
    echo "Stopping current process on port 8187 (PID ${OLD_PID})..."
    kill "$OLD_PID" || true
    while lsof -ti :8187 >/dev/null 2>&1; do
        sleep 0.5
    done
    echo "Port 8187 is now free."
fi

cd "${MYDS_FLEET_DIR:?set MYDS_FLEET_DIR to your deployment directory}"
./start_fleet.sh

echo "Waiting for http://127.0.0.1:8187/health..."
for i in {1..60}; do
    if curl -s http://127.0.0.1:8187/health 2>/dev/null | grep -q '"status":"ok"'; then
        echo "Server is healthy!"
        break
    fi
    sleep 1
done

PID=$(lsof -ti :8187 || true)
if [ -z "$PID" ]; then
    echo "ERROR: Server failed to start on 8187!"
    exit 1
fi

echo "PID=${PID} running on 8187"
echo "--- Environment of running process ---"
ps eww "$PID" | tr ' ' '\n' | grep -E 'MYDS_DENSE_Q4_NR0|MYDS_DENSE_Q4_NR4_MASK|MYDS_MOE_DOWN_NSG' || true

echo "--- Dense Q4 providers in log ---"
grep "Dense Q4 V41 provider" ${MYDS_FLEET_DIR}/logs/local-myds-deepseek-v4.1-flash-1017-q2-q4.log | tail -n 3 || true

echo "--- Overlap status in log ---"
grep "Decode candidates: down overlap=" ${MYDS_FLEET_DIR}/logs/local-myds-deepseek-v4.1-flash-1017-q2-q4.log | tail -n 1 || true
