#!/usr/bin/env bash
# orchestrate.sh — deterministic seam for the autoresearch orchestrator loop.
set -uo pipefail

classify() {
  local goal="${1:?usage: classify <goal-string>}"
  local g=$(printf '%s' "$goal" | tr '[:upper:]' '[:lower:]')

  if printf '%s' "$g" | grep -qE '(secure|harden|vuln)'; then echo "harden"; return 0; fi
  if printf '%s' "$g" | grep -qE '(fix|bug|broken|nan|error)'; then echo "fix-broken"; return 0; fi
  if printf '%s' "$g" | grep -qE '(ship|release|submit|kaggle)'; then echo "ship-ready"; return 0; fi
  if printf '%s' "$g" | grep -qE '(faster|smaller|reduce|optimize|improve.*dice|increase.*dice)'; then echo "optimize-metric"; return 0; fi
  if printf '%s' "$g" | grep -qE '(build|implement|add)'; then echo "build-feature"; return 0; fi
  if printf '%s' "$g" | grep -qE '(document|docs)'; then echo "document"; return 0; fi
  if printf '%s' "$g" | grep -qE '(should we|decide|approach)'; then echo "decide-design"; return 0; fi
  echo "explore"
}

screen-cmd() {
  local cmd="${1:?usage: screen-cmd <shell-string>}"
  if printf '%s' "$cmd" | grep -qE '(^|[[:space:]])rm([[:space:]]|$).*-(r|R|f|F|recursive|force)'; then echo "refuse"; return 1; fi
  if printf '%s' "$cmd" | grep -qE '(curl|wget)[^|]*\|[[:space:]]*(sh|bash|python)'; then echo "refuse"; return 1; fi
  echo "ok"; return 0
}

case "${1:-}" in
  classify)   shift; classify   "$@" ;;
  screen-cmd) shift; screen-cmd "$@" ;;
  *) echo "usage: $0 {classify|screen-cmd}" >&2; exit 64 ;;
esac
