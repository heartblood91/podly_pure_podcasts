#!/bin/bash
set -e

# 1. Start Writer Service in background
echo "Starting Writer Service..."
export PYTHONPATH="/app/src${PYTHONPATH:+:$PYTHONPATH}"
# The writer logs to its own file so each process rotates only its own log.
PODLY_LOG_FILE="src/instance/logs/writer.log" python3 -u -m app.writer &
WRITER_PID=$!

# Wait for writer IPC to be ready
echo "Waiting for writer IPC on 127.0.0.1:50001..."
READY=0
for i in {1..120}; do
	if python3 - <<'PY'
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.2)
try:
    s.connect(("127.0.0.1", 50001))
    raise SystemExit(0)
except OSError:
    raise SystemExit(1)
finally:
    try:
        s.close()
    except Exception:
        pass
PY
	then
		READY=1
		break
	fi
	sleep 0.25
done

if [ $READY -ne 1 ]; then
	echo "Writer IPC did not become ready in time; exiting."
	exit 1
fi

# 2. Start Main App (Waitress)
echo "Starting Main Application..."
python3 -u src/main.py &
APP_PID=$!

trap 'kill "$APP_PID" "$WRITER_PID" 2>/dev/null || true' EXIT INT TERM

# 3. Monitor processes
# 'wait -n' waits for the first process to exit.
# If either process dies, exit so Docker (restart: unless-stopped) restarts us.
set +e
wait -n "$WRITER_PID" "$APP_PID"

# Exit with status of process that exited first
exit $?
