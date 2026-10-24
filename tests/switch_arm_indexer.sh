#!/bin/bash
#
# 部署目录由 MYDS_FLEET_DIR 指定（本仓库不含该目录）。
# 未设置时这些脚本会明确报错退出，而不是猜一个路径。

set -euo pipefail

BATCH_FLOATS="${1:-16777216}"

echo "=== Switching Fleet to MYDS_INDEXER_BATCH_FLOATS=${BATCH_FLOATS} ==="
export MYDS_INDEXER_BATCH_FLOATS="${BATCH_FLOATS}"

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
for i in {1..120}; do
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
ps eww "$PID" | tr ' ' '\n' | grep -E 'MYDS_INDEXER_BATCH_FLOATS' || true
