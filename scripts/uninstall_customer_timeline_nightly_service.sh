#!/usr/bin/env bash
set -euo pipefail

APPLY=0
PLIST_TARGET="${HOME}/Library/LaunchAgents/com.mango.customer-timeline-nightly.plist"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      APPLY=1
      shift
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

echo "Would uninstall launchd plist:"
echo "  launchctl bootout gui/$(id -u) ${PLIST_TARGET}"
echo "  rm -f ${PLIST_TARGET}"

if [[ "${APPLY}" != "1" ]]; then
  echo "Dry-run only. Re-run with --apply after owner approval."
  exit 0
fi

launchctl bootout "gui/$(id -u)" "${PLIST_TARGET}" 2>/dev/null || true
rm -f "${PLIST_TARGET}"
echo "Uninstalled ${PLIST_TARGET}"

