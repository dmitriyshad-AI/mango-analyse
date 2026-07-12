#!/usr/bin/env bash
set -euo pipefail

APPLY=0
PLIST_SOURCE=""
PLIST_TARGET="${HOME}/Library/LaunchAgents/com.mango.customer-timeline-nightly.plist"
CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

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
    --code-root)
      CODE_ROOT="$2"
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
echo "  code root: ${CODE_ROOT}"
echo "  launchctl bootstrap gui/$(id -u) ${PLIST_TARGET}"

if [[ "${APPLY}" != "1" ]]; then
  echo "Dry-run only. Re-run with --apply after owner approval."
  exit 0
fi

install -d -m 0755 "$(dirname "${PLIST_TARGET}")"
python3 - "${PLIST_SOURCE}" "${PLIST_TARGET}" "${CODE_ROOT}" <<'PY'
import os
import plistlib
import sys
from pathlib import Path
from xml.sax.saxutils import escape

source, target, code_root = map(Path, sys.argv[1:])
rendered = source.read_text(encoding="utf-8").replace(
    "__MANGO_CODE_ROOT__", escape(str(code_root.resolve()))
)
if "__MANGO_CODE_ROOT__" in rendered:
    raise SystemExit("unresolved code-root marker")
plistlib.loads(rendered.encode("utf-8"))
temp = target.with_suffix(target.suffix + ".tmp")
temp.write_text(rendered, encoding="utf-8")
os.chmod(temp, 0o644)
temp.replace(target)
PY
launchctl bootstrap "gui/$(id -u)" "${PLIST_TARGET}"
echo "Installed and bootstrapped ${PLIST_TARGET}"
