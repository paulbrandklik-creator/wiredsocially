#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"

echo ""
echo "  ██╗    ██╗██╗██████╗ ███████╗██████╗ "
echo "  ██║    ██║██║██╔══██╗██╔════╝██╔══██╗"
echo "  ██║ █╗ ██║██║██████╔╝█████╗  ██║  ██║"
echo "  ██║███╗██║██║██╔══██╗██╔══╝  ██║  ██║"
echo "  ╚███╔███╔╝██║██║  ██║███████╗██████╔╝"
echo "   ╚══╝╚══╝ ╚═╝╚═╝  ╚═╝╚══════╝╚═════╝ "
echo "  ╔═════════════════════════════════════╗"
echo "  ║   RAG Voice Assistant  —  v1.0      ║"
echo "  ╚═════════════════════════════════════╝"
echo ""

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "  ⚠  ANTHROPIC_API_KEY is not set."
  echo "     Export it first:  export ANTHROPIC_API_KEY=sk-ant-..."
  echo ""
  exit 1
fi

echo "  ✓  ANTHROPIC_API_KEY detected"

# Install Python deps
echo "  →  Checking Python dependencies..."
cd "$BACKEND_DIR"
pip3 install -r requirements.txt -q --disable-pip-version-check

echo "  ✓  Dependencies ready"
echo ""
echo "  →  Starting server at http://localhost:8000"
echo "     Open that URL in Chrome for voice support."
echo ""
echo "  Press Ctrl+C to stop."
echo ""

# Start FastAPI (serves frontend + API)
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
