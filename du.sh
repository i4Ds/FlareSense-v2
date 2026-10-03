#!/usr/bin/env bash
# Docker utility. Same shape as heliotime-v2.
# Usage: ./du.sh <env> <rest>
#   env: dev / prod
#   rest:
#     up -d, logs, stop, ... -> compose subcommands
#     deploy [service]       -> build and start the checked-out version

set -e
set -o pipefail

if [ -t 1 ] && command -v tput >/dev/null 2>&1; then
  bold=$(tput bold || true)
  red=$(tput setaf 1 || true)
  yellow=$(tput setaf 3 || true)
  green=$(tput setaf 2 || true)
  cyan=$(tput setaf 6 || true)
  default=$(tput sgr0 || true)
fi
bold=${bold:-}
red=${red:-}
yellow=${yellow:-}
green=${green:-}
cyan=${cyan:-}
default=${default:-}

if [ -z "$1" ]; then
  echo "${red}No arguments passed.${default}"
  echo "Docs are in the source header. View using: ${cyan}less $0${default}"
  exit 1
fi

env=$1
if [ "$env" != "dev" ] && [ "$env" != "prod" ]; then
  echo "${red}Invalid environment $env.${default} Must be ${cyan}dev${default} or ${cyan}prod${default}."
  exit 1
fi

compose () {
  set -o xtrace
  docker compose --project-directory . --project-name "flaresense-$env" \
    --file ./docker/compose.yml \
    --file "./docker/compose.$env.yml" \
    "$@"
}

if [ "$env" = "dev" ]; then
  mkdir -p "${FLARESENSE_HOST_IMAGE_DIR:-./data/burst_images}"
fi

if [ "$2" = "deploy" ]; then
  compose up --detach --build "${@:3}"
else
  compose "${@:2}"
fi
