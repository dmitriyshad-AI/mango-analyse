#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/git_bootstrap.sh <remote_url> [branch]

Examples:
  scripts/git_bootstrap.sh git@github.com:YOUR_USER/mango-analyse.git
  scripts/git_bootstrap.sh https://github.com/YOUR_USER/mango-analyse.git main
EOF
}

REMOTE_URL="${1:-}"
BRANCH="${2:-main}"

if [[ -z "${REMOTE_URL}" ]]; then
  usage
  exit 2
fi

if [[ ! -d ".git" ]]; then
  git init -b "${BRANCH}"
fi

if ! git config user.name >/dev/null; then
  echo "git user.name is not set. Configure it first:"
  echo "  git config --global user.name \"Your Name\""
  exit 2
fi
if ! git config user.email >/dev/null; then
  echo "git user.email is not set. Configure it first:"
  echo "  git config --global user.email \"you@example.com\""
  exit 2
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "${REMOTE_URL}"
else
  git remote add origin "${REMOTE_URL}"
fi

git branch -M "${BRANCH}"

if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  git add -A
  git commit -m "chore: initial project snapshot"
fi

git push -u origin "${BRANCH}"
echo "Done. Remote 'origin' is set to ${REMOTE_URL}, branch '${BRANCH}' is tracking."
