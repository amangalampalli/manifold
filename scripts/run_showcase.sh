#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-manifold}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

cleanup() {
  if [[ -n "${API_PID:-}" ]] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$ROOT_DIR"

echo "Starting FastAPI backend on http://${API_HOST}:${API_PORT}"
conda run --no-capture-output -n "$CONDA_ENV" \
  uvicorn manifold.api.server:app --host "$API_HOST" --port "$API_PORT" &
API_PID=$!

echo "Starting Vite frontend on http://${FRONTEND_HOST}:${FRONTEND_PORT}"
cd "$ROOT_DIR/frontend"
conda run --no-capture-output -n "$CONDA_ENV" \
  npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT" &
FRONTEND_PID=$!

echo
echo "Showcase running:"
echo "  frontend: http://${FRONTEND_HOST}:${FRONTEND_PORT}"
echo "  backend:  http://${API_HOST}:${API_PORT}"
echo
echo "Press Ctrl-C to stop both."

while true; do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    wait "$API_PID" || true
    exit 1
  fi
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    wait "$FRONTEND_PID" || true
    exit 1
  fi
  sleep 1
done
