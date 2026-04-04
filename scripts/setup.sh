#!/usr/bin/env bash
# AnyCode project setup script
# Usage: bash scripts/setup.sh
set -euo pipefail

echo "=== AnyCode Project Setup ==="
echo ""

# Check Python version
echo "--- Checking Python ---"
PYTHON_VERSION=$(python3 --version 2>/dev/null || python --version 2>/dev/null)
echo "Found: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo ""
    echo "--- Creating virtual environment ---"
    python3 -m venv .venv || python -m venv .venv
    echo "✓ Created .venv"
fi

# Activate
echo ""
echo "--- Activating virtual environment ---"
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null
echo "✓ Activated"

# Install dependencies
echo ""
echo "--- Installing dependencies ---"
if command -v uv &>/dev/null; then
    echo "Using uv..."
    uv sync
else
    echo "Using pip..."
    pip install -e ".[dev]"
fi
echo "✓ Dependencies installed"

# Verify tools
echo ""
echo "--- Verifying tools ---"

if command -v ruff &>/dev/null; then
    echo "✓ ruff $(ruff --version)"
else
    echo "✗ ruff not found — install with: pip install ruff"
fi

if command -v pyright &>/dev/null; then
    echo "✓ pyright available"
else
    echo "✗ pyright not found — install with: pip install pyright or npm install -g pyright"
fi

if command -v pytest &>/dev/null; then
    echo "✓ pytest $(pytest --version | head -1)"
else
    echo "✗ pytest not found"
fi

# Check .env
echo ""
if [ -f ".env" ]; then
    echo "✓ .env file exists"
else
    echo "⚠ No .env file found. Create one with your API keys:"
    echo "  ANTHROPIC_API_KEY=sk-ant-..."
    echo "  OPENAI_API_KEY=sk-..."
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  pytest tests/ -v"
echo "  python examples/01_solo_worker.py"
