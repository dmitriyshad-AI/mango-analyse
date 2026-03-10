# Git + Auto Push + Auto Deploy

This document describes how to set up:

1. Remote Git repository in your account
2. Optional local auto commit/push loop
3. GitHub Actions CI and automatic deploy to your server

## 1) Create remote repo and push current project

1. Create an empty repository in your Git account (for example on GitHub), without README/license.
2. Run:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
git config --global user.name "YOUR_NAME"
git config --global user.email "YOUR_EMAIL"
bash scripts/git_bootstrap.sh git@github.com:YOUR_USER/YOUR_REPO.git main
```

After that, `origin` is configured and `main` is pushed.

## 2) Optional: automatic commit + push every N seconds

Start:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
bash scripts/start_autocommit_push.sh 120 main
```

Stop:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
bash scripts/stop_autocommit_push.sh
```

Log:

```bash
tail -f "/Users/dmitrijfabarisov/Projects/Mango analyse/.git/autocommit_push.log"
```

Notes:
- interval is in seconds (minimum 10);
- if push fails (network/conflict), loop keeps running and retries on next cycle.

## 3) CI and auto deploy are already added

Workflows:

- `.github/workflows/ci.yml`
- `.github/workflows/deploy-server.yml`

Deploy workflow triggers on every push to `main` and:

1. syncs repo to server via `rsync` over SSH,
2. runs `deploy/server_rebuild.sh` on server.

## 4) Configure GitHub Secrets

In repository settings, add:

- `DEPLOY_SSH_PRIVATE_KEY` - private key for SSH
- `DEPLOY_SSH_HOST` - server IP or host
- `DEPLOY_SSH_USER` - SSH user (recommended `deploy`)
- `DEPLOY_SSH_PORT` - optional (default `22`)
- `DEPLOY_APP_DIR` - optional (default `/opt/mango-analyse`)

For your current server:

- host can be `151.242.88.24`
- app dir can be `/opt/mango-analyse`

## 5) Server prerequisites for deploy workflow

On server:

1. SSH user exists and accepts your key.
2. `rsync` is installed:

```bash
sudo apt update
sudo apt install -y rsync
```

3. If you use Docker deployment, Docker is installed and running.

## 6) Customize rebuild logic

Edit:

- `deploy/server_rebuild.sh`

Current default behavior:
- if `docker-compose.yml` or `compose.yml` exists, runs:
  - `docker compose pull`
  - `docker compose up -d --build`

If no compose file exists, script exits with message and no error.
