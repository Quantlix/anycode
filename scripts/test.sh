#!/usr/bin/env bash
# Test runner for anycode-python
# Usage: bash scripts/test.sh [--coverage] [--verbose] [test_path]
set -euo pipefail

COVERAGE=false
VERBOSE=false
TEST_PATH="tests/"

for arg in "$@"; do
    case $arg in
        --coverage) COVERAGE=true ;;
        --verbose)  VERBOSE=true ;;
        *)          TEST_PATH="$arg" ;;
    esac
done

echo "=== AnyCode Tests ==="
echo "Path: $TEST_PATH"
echo ""

PYTEST_ARGS=("$TEST_PATH")

if [ "$VERBOSE" = true ]; then
    PYTEST_ARGS+=("-v" "--tb=short")
else
    PYTEST_ARGS+=("--tb=short")
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_ARGS+=("--cov=src/anycode" "--cov-report=term-missing")
fi

pytest "${PYTEST_ARGS[@]}"
