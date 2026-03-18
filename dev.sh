#!/usr/bin/env bash
# Launch Spaghetti Lab in dev mode.
# Kills any leftover processes first to avoid double windows / port conflicts.
set -euo pipefail

# Kill leftover instances
taskkill //F //IM "spaghetti-lab.exe" 2>/dev/null || true

# Free port 1420 if occupied
PORT_PID=$(netstat -ano 2>/dev/null | grep ":1420 " | grep LISTENING | awk '{print $5}' | head -1)
if [[ -n "$PORT_PID" ]]; then
    echo "Killing process $PORT_PID on port 1420"
    taskkill //F //PID "$PORT_PID" 2>/dev/null || true
    sleep 1
fi

cd "$(dirname "$0")/app"
exec npx tauri dev
