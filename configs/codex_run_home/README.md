# Codex run home

This directory is the repository template for long Mango evaluation runs that invoke `codex exec`.

Default policy:

- use `service_tier = "flex"` for M1 and local measurement runs;
- do not store auth files or secrets here;
- copy local auth into the temporary `CODEX_HOME` only on the machine that runs the measurement;
- use `fast` only as an explicit one-off override for urgent runs.

Example:

```bash
export CODEX_HOME="/private/tmp/mango_codex_home_flex_run"
mkdir -p "$CODEX_HOME"
cp configs/codex_run_home/config.toml "$CODEX_HOME/config.toml"
cp "$HOME/.codex/auth.json" "$CODEX_HOME/" 2>/dev/null || true
```
