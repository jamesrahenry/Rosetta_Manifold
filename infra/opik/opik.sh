#!/usr/bin/env bash
# opik.sh — Manage the self-hosted Opik stack for the Vector project.
#
# Usage:
#   ./infra/opik/opik.sh up       # Start Opik (detached)
#   ./infra/opik/opik.sh down     # Stop and remove containers (keeps volumes)
#   ./infra/opik/opik.sh destroy  # Stop and remove containers + volumes (full reset)
#   ./infra/opik/opik.sh logs     # Tail logs from all services
#   ./infra/opik/opik.sh status   # Show container status
#   ./infra/opik/opik.sh configure # Write ~/.opik.conf pointing at local instance

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yaml"
ENV_FILE="${SCRIPT_DIR}/.env"
OPIK_URL="http://localhost:5173"

# Load .env if present so NGINX_PORT etc. are available in this shell
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' "${ENV_FILE}" | xargs)
fi

NGINX_PORT="${NGINX_PORT:-5173}"
OPIK_URL="http://localhost:${NGINX_PORT}"

cmd="${1:-help}"

case "${cmd}" in
  up)
    echo "▶  Starting Opik self-hosted stack…"
    docker compose \
      --project-name opik \
      --file "${COMPOSE_FILE}" \
      --env-file "${ENV_FILE}" \
      up --detach --remove-orphans
    echo ""
    echo "✅  Opik is starting up."
    echo "    UI  → ${OPIK_URL}"
    echo "    API → ${OPIK_URL}/api"
    echo ""
    echo "    Run './infra/opik/opik.sh configure' to point the opik SDK at this instance."
    ;;

  down)
    echo "⏹  Stopping Opik stack (volumes preserved)…"
    docker compose \
      --project-name opik \
      --file "${COMPOSE_FILE}" \
      --env-file "${ENV_FILE}" \
      down
    ;;

  destroy)
    echo "💣  Destroying Opik stack and ALL volumes (data will be lost)…"
    read -r -p "Are you sure? [y/N] " confirm
    if [[ "${confirm}" =~ ^[Yy]$ ]]; then
      docker compose \
        --project-name opik \
        --file "${COMPOSE_FILE}" \
        --env-file "${ENV_FILE}" \
        down --volumes
      echo "Done."
    else
      echo "Aborted."
    fi
    ;;

  logs)
    docker compose \
      --project-name opik \
      --file "${COMPOSE_FILE}" \
      --env-file "${ENV_FILE}" \
      logs --follow --tail=100
    ;;

  status)
    docker compose \
      --project-name opik \
      --file "${COMPOSE_FILE}" \
      --env-file "${ENV_FILE}" \
      ps
    ;;

  configure)
    echo "⚙️  Configuring opik SDK to use local instance at ${OPIK_URL} …"
    if command -v opik &>/dev/null; then
      # --use_local sets url_override to http://localhost:5173/api/ automatically
      opik configure --use_local --yes
      echo "✅  opik SDK configured (wrote ~/.opik.config)."
    else
      # Write ~/.opik.config manually if the CLI is not installed yet
      CONF_FILE="${HOME}/.opik.config"
      cat > "${CONF_FILE}" <<EOF
[opik]
url_override = ${OPIK_URL}/api/
workspace = default
EOF
      echo "✅  Wrote ${CONF_FILE} (opik CLI not found; SDK will pick this up at runtime)."
    fi
    ;;

  help|*)
    echo "Usage: $0 {up|down|destroy|logs|status|configure}"
    echo ""
    echo "  up         Start the Opik stack in the background"
    echo "  down       Stop containers (data volumes preserved)"
    echo "  destroy    Stop containers and delete all data volumes"
    echo "  logs       Tail logs from all services"
    echo "  status     Show running container status"
    echo "  configure  Point the opik Python SDK at this local instance"
    ;;
esac
