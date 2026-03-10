from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Optional


class MangoMvpGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Mango MVP Local Console")
        self.geometry("1200x820")
        self.minsize(980, 700)

        self.worker_process: Optional[subprocess.Popen[str]] = None
        self._log_queue: queue.SimpleQueue[str] = queue.SimpleQueue()

        self.project_dir = tk.StringVar(value=str(Path.cwd()))
        self.recordings_dir = tk.StringVar(value=str(Path.cwd()))
        self.metadata_csv = tk.StringVar(value="")
        self.database_path = tk.StringVar(value=str(Path.cwd() / "mango_mvp.db"))
        self.export_dir = tk.StringVar(value=str(Path.cwd() / "transcripts"))
        current_python = Path(sys.executable).resolve()
        fallback_backend = Path.cwd() / ".venv-asrbench" / "bin"
        fallback_candidates = [fallback_backend / "python", fallback_backend / "python3"]
        in_known_venv = (
            ".venv-asrbench" in str(current_python)
            or "stable_runtime/venv_stable" in str(current_python)
        )
        if in_known_venv:
            backend_default = current_python
        else:
            backend_default = next(
                (candidate for candidate in fallback_candidates if candidate.exists()),
                current_python,
            )
        self.backend_python = tk.StringVar(value=str(backend_default))
        self.use_project_src = tk.BooleanVar(
            value="stable_runtime/venv_stable" not in str(current_python)
        )

        self.transcribe_provider = tk.StringVar(value="mlx")
        self.dual_enabled = tk.BooleanVar(value=True)
        self.secondary_provider = tk.StringVar(value="gigaam")
        self.merge_provider = tk.StringVar(value="codex_cli")
        self.resolve_llm_provider = tk.StringVar(value="codex_cli")
        self.analyze_provider = tk.StringVar(value="codex_cli")
        self.codex_merge_model = tk.StringVar(value="gpt-5.4")
        self.codex_cli_command = tk.StringVar(value="codex")
        self.codex_cli_timeout_sec = tk.StringVar(value="120")
        self.ollama_base_url = tk.StringVar(value="http://127.0.0.1:11434")
        self.ollama_model = tk.StringVar(value="gpt-oss:20b")
        self.ollama_think = tk.StringVar(value="medium")
        self.ollama_temperature = tk.StringVar(value="0")
        self.split_stereo = tk.BooleanVar(value=True)
        self.language = tk.StringVar(value="ru")
        self.mlx_model = tk.StringVar(value="mlx-community/whisper-large-v3-mlx")
        self.gigaam_model = tk.StringVar(value="v2_rnnt")
        self.gigaam_device = tk.StringVar(value="cpu")
        self.gigaam_segment_sec = tk.StringVar(value="20")
        self.merge_similarity_threshold = tk.StringVar(value="0.985")
        self.mono_role_assignment_mode = tk.StringVar(value="off")
        self.mono_role_assignment_min_confidence = tk.StringVar(value="0.62")
        self.mono_role_assignment_llm_threshold = tk.StringVar(value="0.72")
        self.openai_role_assign_model = tk.StringVar(value="gpt-4o-mini")
        self.stage_limit = tk.StringVar(value="100")

        self.transcribe_max_attempts = tk.StringVar(value="3")
        self.analyze_max_attempts = tk.StringVar(value="3")
        self.sync_max_attempts = tk.StringVar(value="3")
        self.retry_base_delay_sec = tk.StringVar(value="30")
        self.worker_poll_sec = tk.StringVar(value="10")
        self.worker_max_idle_cycles = tk.StringVar(value="30")

        self._build_ui()
        self.after(80, self._drain_log_queue)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        config_frame = ttk.LabelFrame(root, text="Paths")
        config_frame.pack(fill=tk.X, padx=4, pady=4)
        self._path_row(
            config_frame,
            row=0,
            label="Project dir",
            var=self.project_dir,
            ask_func=self._pick_project_dir,
        )
        self._path_row(
            config_frame,
            row=1,
            label="Recordings dir",
            var=self.recordings_dir,
            ask_func=self._pick_recordings_dir,
        )
        self._path_row(
            config_frame,
            row=2,
            label="Metadata CSV",
            var=self.metadata_csv,
            ask_func=self._pick_metadata_csv,
        )
        self._path_row(
            config_frame,
            row=3,
            label="Database file",
            var=self.database_path,
            ask_func=self._pick_database_file,
        )
        self._path_row(
            config_frame,
            row=4,
            label="Transcript export dir",
            var=self.export_dir,
            ask_func=self._pick_export_dir,
        )
        self._path_row(
            config_frame,
            row=5,
            label="Backend Python",
            var=self.backend_python,
            ask_func=self._pick_backend_python,
        )
        ttk.Checkbutton(
            config_frame,
            text="Use project src (dev mode)",
            variable=self.use_project_src,
        ).grid(row=6, column=1, sticky="w", padx=6, pady=3)

        options = ttk.LabelFrame(root, text="ASR + Processing options")
        options.pack(fill=tk.X, padx=4, pady=4)

        self._combo_row(
            options,
            0,
            "Transcribe provider",
            self.transcribe_provider,
            ["mlx", "gigaam", "openai", "mock"],
        )
        ttk.Checkbutton(options, text="Dual transcribe", variable=self.dual_enabled).grid(
            row=0, column=2, sticky="w", padx=6, pady=3
        )
        self._combo_row(
            options,
            1,
            "Secondary provider",
            self.secondary_provider,
            ["gigaam", "mlx", "openai", "mock", ""],
        )
        self._combo_row(
            options,
            2,
            "Merge provider",
            self.merge_provider,
            ["rule", "codex_cli", "ollama", "openai", "primary"],
        )
        self._combo_row(
            options,
            3,
            "Analyze provider",
            self.analyze_provider,
            ["mock", "codex_cli", "ollama", "openai"],
        )
        ttk.Checkbutton(options, text="Split stereo channels", variable=self.split_stereo).grid(
            row=3, column=2, sticky="w", padx=6, pady=3
        )
        self._entry_row(options, 4, "Language", self.language)
        self._entry_row(options, 5, "MLX model", self.mlx_model)
        self._entry_row(options, 6, "GigaAM model", self.gigaam_model)
        self._entry_row(options, 7, "GigaAM device", self.gigaam_device)
        self._entry_row(options, 8, "GigaAM segment sec", self.gigaam_segment_sec)
        self._entry_row(
            options, 9, "Dual merge similarity threshold", self.merge_similarity_threshold
        )
        self._combo_row(
            options,
            10,
            "Mono role assignment",
            self.mono_role_assignment_mode,
            ["off", "rule", "ollama_selective", "openai_selective"],
        )
        self._entry_row(
            options, 11, "Mono assign min confidence", self.mono_role_assignment_min_confidence
        )
        self._entry_row(
            options, 12, "Mono assign LLM threshold", self.mono_role_assignment_llm_threshold
        )
        self._entry_row(options, 13, "OpenAI role-assign model", self.openai_role_assign_model)
        self._combo_row(
            options,
            14,
            "Resolve LLM provider",
            self.resolve_llm_provider,
            ["codex_cli", "ollama", "openai", "off"],
        )
        self._entry_row(options, 15, "Codex merge model", self.codex_merge_model)
        self._entry_row(options, 16, "Codex command", self.codex_cli_command)
        self._entry_row(options, 17, "Codex timeout sec", self.codex_cli_timeout_sec)
        self._entry_row(options, 18, "Ollama base URL", self.ollama_base_url)
        self._entry_row(options, 19, "Ollama model", self.ollama_model)
        self._entry_row(options, 20, "Ollama think (low/medium/high)", self.ollama_think)
        self._entry_row(options, 21, "Ollama temperature", self.ollama_temperature)
        self._entry_row(options, 22, "Stage limit", self.stage_limit)

        reliability = ttk.LabelFrame(root, text="Reliability")
        reliability.pack(fill=tk.X, padx=4, pady=4)
        self._entry_row(reliability, 0, "Transcribe max attempts", self.transcribe_max_attempts)
        self._entry_row(reliability, 1, "Analyze max attempts", self.analyze_max_attempts)
        self._entry_row(reliability, 2, "Sync max attempts", self.sync_max_attempts)
        self._entry_row(reliability, 3, "Retry base delay sec", self.retry_base_delay_sec)
        self._entry_row(reliability, 4, "Worker poll sec", self.worker_poll_sec)
        self._entry_row(reliability, 5, "Worker max idle cycles", self.worker_max_idle_cycles)

        actions = ttk.LabelFrame(root, text="Actions")
        actions.pack(fill=tk.X, padx=4, pady=4)
        for idx, (title, callback) in enumerate(
            [
                ("Init DB", self.on_init_db),
                ("Ingest", self.on_ingest),
                ("Transcribe", self.on_transcribe),
                ("Resolve", self.on_resolve),
                ("Analyze", self.on_analyze),
                ("Sync", self.on_sync),
                ("Export review queue", self.on_export_review_queue),
                ("Export failed resolve", self.on_export_failed_resolve_queue),
                ("Stats", self.on_stats),
                ("Requeue dead", self.on_requeue_dead),
                ("Reset missing variants", self.on_reset_missing_variants),
                ("Worker once", self.on_worker_once),
                ("Worker start (background)", self.on_worker_start),
                ("Worker stop", self.on_worker_stop),
            ]
        ):
            ttk.Button(actions, text=title, command=callback).grid(
                row=0, column=idx, padx=4, pady=6, sticky="ew"
            )

        log_frame = ttk.LabelFrame(root, text="Output")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.log = tk.Text(log_frame, wrap=tk.WORD)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.configure(yscrollcommand=scroll.set)

    def _path_row(self, frame: ttk.LabelFrame, row: int, label: str, var: tk.StringVar, ask_func) -> None:
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(frame, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(frame, text="Browse", command=ask_func).grid(
            row=row, column=2, sticky="ew", padx=6, pady=3
        )
        frame.columnconfigure(1, weight=1)

    def _entry_row(self, frame: ttk.LabelFrame, row: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(frame, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        frame.columnconfigure(1, weight=1)

    def _combo_row(
        self, frame: ttk.LabelFrame, row: int, label: str, var: tk.StringVar, values: list[str]
    ) -> None:
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=3)
        combo = ttk.Combobox(frame, textvariable=var, values=values, state="readonly")
        combo.grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        frame.columnconfigure(1, weight=1)

    def _pick_project_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=self.project_dir.get() or str(Path.cwd()))
        if path:
            self.project_dir.set(path)

    def _pick_recordings_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=self.recordings_dir.get() or str(Path.cwd()))
        if path:
            self.recordings_dir.set(path)

    def _pick_metadata_csv(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=self.project_dir.get() or str(Path.cwd()),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.metadata_csv.set(path)

    def _pick_database_file(self) -> None:
        path = filedialog.asksaveasfilename(
            initialdir=self.project_dir.get() or str(Path.cwd()),
            defaultextension=".db",
            filetypes=[("SQLite DB", "*.db"), ("All files", "*.*")],
        )
        if path:
            self.database_path.set(path)

    def _pick_export_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=self.export_dir.get() or str(Path.cwd()))
        if path:
            self.export_dir.set(path)

    def _pick_backend_python(self) -> None:
        current = self.backend_python.get().strip()
        initial = str(Path(current).parent) if current else (self.project_dir.get() or str(Path.cwd()))
        path = filedialog.askopenfilename(
            initialdir=initial,
            filetypes=[("Python executable", "python*"), ("All files", "*.*")],
        )
        if path:
            self.backend_python.set(path)

    def _bool(self, value: bool) -> str:
        return "1" if value else "0"

    def _backend_python_exe(self) -> str:
        candidate = self.backend_python.get().strip()
        if candidate:
            return candidate
        return sys.executable

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.use_project_src.get():
            env["PYTHONPATH"] = str(Path(self.project_dir.get()) / "src")
        else:
            env.pop("PYTHONPATH", None)
        env["PATH"] = f"{Path(self.project_dir.get()) / '.local' / 'bin'}:{env.get('PATH', '')}"
        env["DATABASE_URL"] = f"sqlite:///{Path(self.database_path.get()).expanduser().resolve()}"
        env["TRANSCRIPT_EXPORT_DIR"] = self.export_dir.get().strip()
        env["TRANSCRIBE_PROVIDER"] = self.transcribe_provider.get().strip()
        env["DUAL_TRANSCRIBE_ENABLED"] = self._bool(self.dual_enabled.get())
        env["SECONDARY_TRANSCRIBE_PROVIDER"] = self.secondary_provider.get().strip()
        env["DUAL_MERGE_PROVIDER"] = self.merge_provider.get().strip()
        env["RESOLVE_LLM_PROVIDER"] = self.resolve_llm_provider.get().strip()
        env["DUAL_MERGE_SIMILARITY_THRESHOLD"] = self.merge_similarity_threshold.get().strip()
        env["CODEX_MERGE_MODEL"] = self.codex_merge_model.get().strip()
        env["CODEX_CLI_COMMAND"] = self.codex_cli_command.get().strip()
        env["CODEX_CLI_TIMEOUT_SEC"] = self.codex_cli_timeout_sec.get().strip()
        env["ANALYZE_PROVIDER"] = self.analyze_provider.get().strip()
        env["SPLIT_STEREO_CHANNELS"] = self._bool(self.split_stereo.get())
        env["TRANSCRIBE_LANGUAGE"] = self.language.get().strip()
        env["MLX_WHISPER_MODEL"] = self.mlx_model.get().strip()
        env["MLX_CONDITION_ON_PREVIOUS_TEXT"] = "false"
        env["MLX_WORD_TIMESTAMPS"] = "true"
        env["GIGAAM_MODEL"] = self.gigaam_model.get().strip()
        env["GIGAAM_DEVICE"] = self.gigaam_device.get().strip()
        env["GIGAAM_SEGMENT_SEC"] = self.gigaam_segment_sec.get().strip()
        env["MONO_ROLE_ASSIGNMENT_MODE"] = self.mono_role_assignment_mode.get().strip()
        env["MONO_ROLE_ASSIGNMENT_MIN_CONFIDENCE"] = (
            self.mono_role_assignment_min_confidence.get().strip()
        )
        env["MONO_ROLE_ASSIGNMENT_LLM_THRESHOLD"] = (
            self.mono_role_assignment_llm_threshold.get().strip()
        )
        env["OPENAI_ROLE_ASSIGN_MODEL"] = self.openai_role_assign_model.get().strip()
        env["OLLAMA_BASE_URL"] = self.ollama_base_url.get().strip()
        env["OLLAMA_MODEL"] = self.ollama_model.get().strip()
        env["OLLAMA_THINK"] = self.ollama_think.get().strip()
        env["OLLAMA_TEMPERATURE"] = self.ollama_temperature.get().strip()
        env["TRANSCRIBE_MAX_ATTEMPTS"] = self.transcribe_max_attempts.get().strip()
        env["ANALYZE_MAX_ATTEMPTS"] = self.analyze_max_attempts.get().strip()
        env["SYNC_MAX_ATTEMPTS"] = self.sync_max_attempts.get().strip()
        env["RETRY_BASE_DELAY_SEC"] = self.retry_base_delay_sec.get().strip()
        env["WORKER_POLL_SEC"] = self.worker_poll_sec.get().strip()
        env["WORKER_MAX_IDLE_CYCLES"] = self.worker_max_idle_cycles.get().strip()
        return env

    def _append_log(self, text: str) -> None:
        self._log_queue.put(text.rstrip())

    def _drain_log_queue(self) -> None:
        try:
            while True:
                line = self._log_queue.get_nowait()
                self.log.insert(tk.END, line + "\n")
                self.log.see(tk.END)
        except queue.Empty:
            pass
        except tk.TclError:
            return
        self.after(80, self._drain_log_queue)

    def _start_command_thread(self, title: str, args: list[str]) -> None:
        def _runner() -> None:
            self._append_log(f"\n=== {title} ===")
            cmd = [self._backend_python_exe(), "-u", "-m", "mango_mvp.cli", *args]
            self._append_log(f"$ {' '.join(cmd)}")
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=self.project_dir.get(),
                    env=self._build_env(),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if completed.stdout:
                    self._append_log(completed.stdout)
                if completed.stderr:
                    self._append_log(completed.stderr)
                self._append_log(f"[exit_code={completed.returncode}]")
            except Exception as exc:  # noqa: BLE001
                self._append_log(f"[error] {exc}")

        threading.Thread(target=_runner, daemon=True).start()

    def _stage_limit(self) -> str:
        return self.stage_limit.get().strip() or "100"

    def on_init_db(self) -> None:
        self._start_command_thread("init-db", ["init-db"])

    def on_ingest(self) -> None:
        args = ["ingest", "--recordings-dir", self.recordings_dir.get().strip()]
        if self.metadata_csv.get().strip():
            args.extend(["--metadata-csv", self.metadata_csv.get().strip()])
        self._start_command_thread("ingest", args)

    def on_transcribe(self) -> None:
        self._start_command_thread("transcribe", ["transcribe", "--limit", self._stage_limit()])

    def on_analyze(self) -> None:
        self._start_command_thread("analyze", ["analyze", "--limit", self._stage_limit()])

    def on_resolve(self) -> None:
        self._start_command_thread("resolve", ["resolve", "--limit", self._stage_limit()])

    def on_sync(self) -> None:
        self._start_command_thread("sync", ["sync", "--limit", self._stage_limit()])

    def on_export_review_queue(self) -> None:
        out_path = Path(self.project_dir.get()) / "manual_review_queue.csv"
        self._start_command_thread(
            "export-review-queue",
            ["export-review-queue", "--out", str(out_path), "--limit", "100000"],
        )

    def on_export_failed_resolve_queue(self) -> None:
        out_path = Path(self.project_dir.get()) / "failed_resolve_queue.csv"
        self._start_command_thread(
            "export-failed-resolve-queue",
            ["export-failed-resolve-queue", "--out", str(out_path), "--limit", "100000"],
        )

    def on_stats(self) -> None:
        self._start_command_thread("stats", ["stats"])

    def on_requeue_dead(self) -> None:
        self._start_command_thread("requeue-dead", ["requeue-dead", "--stage", "all"])

    def on_reset_missing_variants(self) -> None:
        self._start_command_thread(
            "reset-transcribe missing variants",
            [
                "reset-transcribe",
                "--only-done",
                "--only-missing-variants",
                "--limit",
                self._stage_limit(),
            ],
        )

    def on_worker_once(self) -> None:
        self._start_command_thread(
            "worker once",
            ["worker", "--once", "--stage-limit", self._stage_limit()],
        )

    def on_worker_start(self) -> None:
        if self.worker_process and self.worker_process.poll() is None:
            self._append_log("[worker] already running")
            return
        cmd = [
            self._backend_python_exe(),
            "-u",
            "-m",
            "mango_mvp.cli",
            "worker",
            "--stage-limit",
            self._stage_limit(),
            "--poll-sec",
            self.worker_poll_sec.get().strip() or "10",
            "--max-idle-cycles",
            self.worker_max_idle_cycles.get().strip() or "30",
        ]
        self._append_log("\n=== worker start ===")
        self._append_log(f"$ {' '.join(cmd)}")
        try:
            self.worker_process = subprocess.Popen(
                cmd,
                cwd=self.project_dir.get(),
                env=self._build_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"[worker start error] {exc}")
            self.worker_process = None
            return

        def _reader() -> None:
            assert self.worker_process is not None
            if self.worker_process.stdout:
                for line in self.worker_process.stdout:
                    self._append_log(line.rstrip())
            code = self.worker_process.wait()
            self._append_log(f"[worker exited code={code}]")

        threading.Thread(target=_reader, daemon=True).start()

    def on_worker_stop(self) -> None:
        if not self.worker_process or self.worker_process.poll() is not None:
            self._append_log("[worker] not running")
            return
        self.worker_process.terminate()
        self._append_log("[worker] terminate requested")


def main() -> int:
    app = MangoMvpGui()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
