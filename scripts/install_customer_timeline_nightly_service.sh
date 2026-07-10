#!/usr/bin/env bash
set -euo pipefail

APPLY=0
PLIST_SOURCE=""
PLIST_TARGET="${HOME}/Library/LaunchAgents/com.mango.customer-timeline-nightly.plist"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      APPLY=1
      shift
      ;;
    --plist)
      PLIST_SOURCE="$2"
      shift 2
      ;;
    --target)
      PLIST_TARGET="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${PLIST_SOURCE}" ]]; then
  echo "Usage: $0 --plist <plist> [--target <launch-agent-plist>] [--apply]" >&2
  exit 2
fi

echo "Would install launchd plist:"
echo "  source: ${PLIST_SOURCE}"
echo "  target: ${PLIST_TARGET}"
echo "  launchctl bootstrap gui/$(id -u) ${PLIST_TARGET}"

if [[ "${APPLY}" != "1" ]]; then
  echo "Dry-run only. Re-run with --apply after owner approval."
  exit 0
fi

install -d -m 0755 "$(dirname "${PLIST_TARGET}")"
install -m 0644 "${PLIST_SOURCE}" "${PLIST_TARGET}"
launchctl bootstrap "gui/$(id -u)" "${PLIST_TARGET}"
echo "Installed and bootstrapped ${PLIST_TARGET}"
