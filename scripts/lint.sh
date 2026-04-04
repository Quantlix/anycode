#!/usr/bin/env bash
# Lint and format check for anycode-python
# Usage: bash scripts/lint.sh [--fix]
set -euo pipefail

FIX_MODE=false
if [ "${1:-}" = "--fix" ]; then
    FIX_MODE=true
fi

echo "=== AnyCode Lint ==="
EXIT_CODE=0

# Step 1: Ruff check
echo ""
echo "--- Ruff Check ---"
if [ "$FIX_MODE" = true ]; then
    ruff check src/ --fix && echo "✓ Ruff check passed (with auto-fix)" || EXIT_CODE=1
else
    ruff check src/ && echo "✓ Ruff check passed" || EXIT_CODE=1
fi

# Step 2: Format
echo ""
echo "--- Ruff Format ---"
if [ "$FIX_MODE" = true ]; then
    ruff format src/ && echo "✓ Formatted" || EXIT_CODE=1
else
    ruff format --check src/ && echo "✓ Formatting is correct" || { echo "Run with --fix to auto-format"; EXIT_CODE=1; }
fi

# Step 3: Type check
echo ""
echo "--- Pyright ---"
pyright && echo "✓ Type check passed" || EXIT_CODE=1

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=== All Checks Passed ==="
else
    echo "=== Some Checks Failed ==="
fi

exit $EXIT_CODE
