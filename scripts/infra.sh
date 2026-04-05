#!/usr/bin/env bash
# Docker infrastructure lifecycle for anycode test backends.
# Usage: bash scripts/infra.sh [up|down|status|logs|reset]
set -euo pipefail

CMD="${1:-status}"
COMPOSE_FILE="docker-compose.yml"

case "$CMD" in
    up)
        echo "Starting test infrastructure…"
        docker compose -f "$COMPOSE_FILE" up -d --wait
        echo "All services healthy."
        ;;
    down)
        echo "Stopping test infrastructure…"
        docker compose -f "$COMPOSE_FILE" down -v --remove-orphans
        echo "Done."
        ;;
    status)
        docker compose -f "$COMPOSE_FILE" ps
        ;;
    logs)
        docker compose -f "$COMPOSE_FILE" logs --tail=50 -f
        ;;
    reset)
        echo "Resetting test infrastructure…"
        docker compose -f "$COMPOSE_FILE" down -v --remove-orphans
        docker compose -f "$COMPOSE_FILE" up -d --wait
        echo "All services healthy."
        ;;
    *)
        echo "Usage: $0 {up|down|status|logs|reset}"
        exit 1
        ;;
esac
