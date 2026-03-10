from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any, Dict, Optional


STATUS_REFRESH_SEC = 12
LOG_POLL_MS = 100


class MangoMvpGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Mango Calls Studio")
        self.geometry("1320x920")
        self.minsize(1080, 760)

        self.worker_process: Optional[subprocess.Popen[str]] = None
        self.current_process: Optional[subprocess.Popen[str]] = None
        self.current_process_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.active_task: Optional[threading.Thread] = None
        self._log_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._auto_stats_job: Optional[str] = None

        self.project_dir = tk.StringVar(value=str(Path.cwd()))
        self.recordings_dir = tk.StringVar(value=str(Path.cwd()))
        self.metadata_csv = tk.StringVar(value="")
        self.database_path = tk.StringVar(value=str(Path.cwd() / "mango_mvp.db"))
        self.export_dir = tk.StringVar(value=str(Path.cwd() / "transcripts"))

        current_python = Path(sys.executable).resolve()
        stable_backend = Path.cwd() / "stable_runtime" / "venv_stable" / "bin"
        asrbench_backend = Path.cwd() / ".venv-asrbench" / "bin"
        fallback_candidates = [
            current_python,
            stable_backend / "python",
            stable_backend / "python3",
            asrbench_backend / "python",
            asrbench_backend / "python3",
        ]
        backend_default = next(
            (candidate for candidate in fallback_candidates if self._python_has_runtime_deps(candidate)),
            current_python,
        )
        self.backend_python = tk.StringVar(value=str(backend_default))
        self.use_project_src = tk.BooleanVar(
            value="stable_runtime/venv_stable" not in str(backend_default)
        )

        self.transcribe_mode = tk.StringVar(value="dual")
        self.transcribe_provider = tk.StringVar(value="mlx")
        self.secondary_provider = tk.StringVar(value="gigaam")
        self.dual_mode_state = tk.StringVar(value="")
        self.merge_provider = tk.StringVar(value="codex_cli")
        self.language = tk.StringVar(value="ru")
        self.stage_limit = tk.StringVar(value="100")

        self.split_stereo = tk.BooleanVar(value=True)
        self.mono_role_assignment_mode = tk.StringVar(value="rule")
        self.mono_role_assignment_min_confidence = tk.StringVar(value="0.62")
        self.mono_role_assignment_llm_threshold = tk.StringVar(value="0.72")
        self.openai_role_assign_model = tk.StringVar(value="gpt-4o-mini")

        self.mlx_model = tk.StringVar(value="mlx-community/whisper-large-v3-mlx")
        self.gigaam_model = tk.StringVar(value="v2_rnnt")
        self.gigaam_device = tk.StringVar(value="cpu")
        self.gigaam_segment_sec = tk.StringVar(value="20")
        self.merge_similarity_threshold = tk.StringVar(value="0.985")

        self.resolve_llm_provider = tk.StringVar(value="codex_cli")
        self.analyze_provider = tk.StringVar(value="codex_cli")
        self.codex_merge_model = tk.StringVar(value="gpt-5.4")
        self.codex_cli_command = tk.StringVar(value="codex")
        self.codex_cli_timeout_sec = tk.StringVar(value="180")

        self.ollama_base_url = tk.StringVar(value="http://127.0.0.1:11434")
        self.ollama_model = tk.StringVar(value="gpt-oss:20b")
        self.ollama_think = tk.StringVar(value="medium")
        self.ollama_temperature = tk.StringVar(value="0")

        self.transcribe_max_attempts = tk.StringVar(value="3")
        self.analyze_max_attempts = tk.StringVar(value="3")
        self.sync_max_attempts = tk.StringVar(value="3")
        self.retry_base_delay_sec = tk.StringVar(value="30")
        self.worker_poll_sec = tk.StringVar(value="10")
        self.worker_max_idle_cycles = tk.StringVar(value="30")

        self.run_state = tk.StringVar(value="Idle")
        self.metric_vars = {
            "total_calls": tk.StringVar(value="0"),
            "tr_done": tk.StringVar(value="0"),
            "tr_pending": tk.StringVar(value="0"),
            "tr_failed": tk.StringVar(value="0"),
            "tr_dead": tk.StringVar(value="0"),
            "rs_done": tk.StringVar(value="0"),
            "rs_skipped": tk.StringVar(value="0"),
            "rs_manual": tk.StringVar(value="0"),
            "rs_pending": tk.StringVar(value="0"),
            "an_done": tk.StringVar(value="0"),
            "an_pending": tk.StringVar(value="0"),
            "an_failed": tk.StringVar(value="0"),
            "an_dead": tk.StringVar(value="0"),
        }

        self._setup_theme()
        self.transcribe_mode.trace_add("write", self._on_transcribe_mode_changed)
        self._on_transcribe_mode_changed()
        self._build_ui()
        self.after(LOG_POLL_MS, self._drain_log_queue)
        self._refresh_stats_async(silent=True)
        self._schedule_auto_stats()

    @staticmethod
    def _python_has_runtime_deps(executable: Path) -> bool:
        if not executable.exists():
            return False
        try:
            proc = subprocess.run(
                [str(executable), "-c", "import sqlalchemy, dotenv"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
            return proc.returncode == 0
        except Exception:  # noqa: BLE001
            return False

    def _setup_theme(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        bg = "#F2F4F7"
        card = "#FFFFFF"
        primary = "#0A84FF"
        text = "#111827"
        muted = "#6B7280"
        line = "#D1D5DB"

        self.configure(bg=bg)
        style.configure("App.TFrame", background=bg)
        style.configure("Card.TLabelframe", background=card, borderwidth=1, relief=tk.SOLID)
        style.configure("Card.TLabelframe.Label", background=card, foreground=text)
        style.configure("App.TLabel", background=bg, foreground=text)
        style.configure("Muted.TLabel", background=bg, foreground=muted)
        style.configure("CardValue.TLabel", background=card, foreground=text, font=self._font(18, "bold"))
        style.configure("CardCaption.TLabel", background=card, foreground=muted, font=self._font(10, "normal"))
        style.configure("Header.TLabel", background=bg, foreground=text, font=self._font(22, "bold"))
        style.configure("SubHeader.TLabel", background=bg, foreground=muted, font=self._font(11, "normal"))
        style.configure("Primary.TButton", foreground="#FFFFFF")
        style.map(
            "Primary.TButton",
            background=[("active", "#006FE8"), ("!disabled", primary)],
            foreground=[("disabled", "#D1D5DB"), ("!disabled", "#FFFFFF")],
        )
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(12, 8))
        style.configure("Thin.Horizontal.TProgressbar", thickness=8, troughcolor=line, background=primary)

    def _font(self, size: int, weight: str) -> tuple[str, int, str]:
        families = set(tkfont.families())
        preferred = [
            "SF Pro Text",
            "SF Pro Display",
            "Helvetica Neue",
            "Helvetica",
            "Arial",
        ]
        family = next((name for name in preferred if name in families), "TkDefaultFont")
        return (family, size, weight)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, style="App.TFrame", padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root, style="App.TFrame")
        header.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(header, text="Mango Calls Studio", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "Stable local cockpit: transcribe queue, Codex post-processing queue, and resume-safe runs "
                "(done calls are never reprocessed unless you reset statuses manually)."
            ),
            style="SubHeader.TLabel",
            wraplength=1180,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(2, 6))

        status_row = ttk.Frame(header, style="App.TFrame")
        status_row.pack(fill=tk.X)
        ttk.Label(status_row, text="Run state:", style="Muted.TLabel").pack(side=tk.LEFT)
        ttk.Label(status_row, textvariable=self.run_state, style="App.TLabel").pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(
            status_row,
            text="Stop Current Task",
            command=self.on_stop_current_task,
            style="Primary.TButton",
        ).pack(side=tk.RIGHT)

        top = ttk.Panedwindow(root, orient=tk.VERTICAL)
        top.pack(fill=tk.BOTH, expand=True)

        notebook_wrap = ttk.Frame(top, style="App.TFrame")
        logs_wrap = ttk.Frame(top, style="App.TFrame")
        top.add(notebook_wrap, weight=4)
        top.add(logs_wrap, weight=2)

        notebook = ttk.Notebook(notebook_wrap)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_pipeline = ttk.Frame(notebook, style="App.TFrame")
        self.tab_status = ttk.Frame(notebook, style="App.TFrame")
        self.tab_functions = ttk.Frame(notebook, style="App.TFrame")
        self.tab_design = ttk.Frame(notebook, style="App.TFrame")
        notebook.add(self.tab_pipeline, text="Pipeline")
        notebook.add(self.tab_status, text="Status")
        notebook.add(self.tab_functions, text="Functions")
        notebook.add(self.tab_design, text="Design Review")

        self._build_pipeline_tab(self.tab_pipeline)
        self._build_status_tab(self.tab_status)
        self._build_functions_tab(self.tab_functions)
        self._build_design_tab(self.tab_design)
        self._build_log_panel(logs_wrap)

    def _on_transcribe_mode_changed(self, *_: object) -> None:
        mode = self.transcribe_mode.get().strip().lower()
        if mode == "dual":
            self.dual_mode_state.set("Dual mode: ON (two ASR systems)")
            if not self.secondary_provider.get().strip():
                self.secondary_provider.set("gigaam")
        elif mode == "whisper":
            self.dual_mode_state.set("Dual mode: OFF (Whisper only)")
        elif mode == "gigaam":
            self.dual_mode_state.set("Dual mode: OFF (GigaAM only)")
        else:
            self.dual_mode_state.set("Dual mode: OFF")

    def _make_scrollable_tab(self, parent: ttk.Frame) -> ttk.Frame:
        wrap = ttk.Frame(parent, style="App.TFrame")
        wrap.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(wrap, bg="#F2F4F7", highlightthickness=0, borderwidth=0)
        vbar = ttk.Scrollbar(wrap, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)

        content = ttk.Frame(canvas, style="App.TFrame")
        win_id = canvas.create_window((0, 0), window=content, anchor="nw")

        def _sync_scroll_region(_event: object) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_width(_event: object) -> None:
            canvas.itemconfigure(win_id, width=canvas.winfo_width())

        def _on_mousewheel(event: tk.Event) -> None:
            if hasattr(event, "delta") and event.delta:
                # macOS trackpad/mouse
                step = int(-event.delta / 120) if event.delta % 120 == 0 else int(-event.delta / 6)
                if step == 0:
                    step = -1 if event.delta > 0 else 1
                canvas.yview_scroll(step, "units")
                return
            if getattr(event, "num", None) == 4:
                canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(1, "units")

        def _bind_wheel(_event: object) -> None:
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_wheel(_event: object) -> None:
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        content.bind("<Configure>", _sync_scroll_region)
        canvas.bind("<Configure>", _sync_width)
        wrap.bind("<Enter>", _bind_wheel)
        wrap.bind("<Leave>", _unbind_wheel)
        return content

    def _build_pipeline_tab(self, parent: ttk.Frame) -> None:
        content = self._make_scrollable_tab(parent)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)

        paths_card = ttk.LabelFrame(content, text="Project & Paths", style="Card.TLabelframe", padding=10)
        paths_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        self._path_row(paths_card, 0, "Project dir", self.project_dir, self._pick_project_dir)
        self._path_row(paths_card, 1, "Recordings dir", self.recordings_dir, self._pick_recordings_dir)
        self._path_row(paths_card, 2, "Metadata CSV (optional)", self.metadata_csv, self._pick_metadata_csv)
        self._path_row(paths_card, 3, "Database file", self.database_path, self._pick_database_file)
        self._path_row(paths_card, 4, "Transcript export dir", self.export_dir, self._pick_export_dir)
        self._path_row(paths_card, 5, "Backend Python", self.backend_python, self._pick_backend_python)
        ttk.Checkbutton(
            paths_card,
            text="Use project src (dev mode). Disable for stable snapshot runtime.",
            variable=self.use_project_src,
        ).grid(row=6, column=1, sticky="w", padx=6, pady=4)

        asr_card = ttk.LabelFrame(content, text="Stage 1: Transcription Setup", style="Card.TLabelframe", padding=10)
        asr_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))

        self._combo_row(
            asr_card,
            0,
            "Transcribe mode",
            self.transcribe_mode,
            ["dual", "whisper", "gigaam"],
        )
        self._combo_row(
            asr_card,
            1,
            "Primary provider",
            self.transcribe_provider,
            ["mlx", "gigaam", "openai", "mock"],
        )
        self._combo_row(
            asr_card,
            2,
            "Secondary provider",
            self.secondary_provider,
            ["gigaam", "mlx", "openai", "mock", ""],
        )
        ttk.Label(
            asr_card,
            textvariable=self.dual_mode_state,
            style="Muted.TLabel",
        ).grid(
            row=2,
            column=2,
            sticky="w",
            padx=6,
            pady=3,
        )
        self._combo_row(
            asr_card,
            3,
            "Dual merge provider",
            self.merge_provider,
            ["rule", "codex_cli", "ollama", "openai", "primary"],
        )
        ttk.Label(asr_card, text="Records per batch (--limit)", style="App.TLabel").grid(
            row=4, column=0, sticky="w", padx=6, pady=3
        )
        ttk.Spinbox(
            asr_card,
            from_=1,
            to=10000,
            increment=10,
            textvariable=self.stage_limit,
            width=12,
        ).grid(row=4, column=1, sticky="w", padx=6, pady=3)
        preset = ttk.Frame(asr_card, style="App.TFrame")
        preset.grid(row=4, column=2, sticky="w", padx=6, pady=3)
        ttk.Label(preset, text="Quick:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        for value in (50, 100, 250, 500):
            ttk.Button(
                preset,
                text=str(value),
                command=lambda v=value: self.stage_limit.set(str(v)),
                width=5,
            ).pack(side=tk.LEFT, padx=1)

        ttk.Label(
            asr_card,
            text=(
                "To run only recognition: choose mode and click 'Stage 1: Transcribe All Pending'. "
                "Batch count is this field."
            ),
            style="Muted.TLabel",
            wraplength=440,
            justify=tk.LEFT,
        ).grid(row=5, column=0, columnspan=3, sticky="w", padx=6, pady=(2, 6))

        self._entry_row(asr_card, 6, "Language", self.language)
        self._entry_row(asr_card, 7, "MLX Whisper model", self.mlx_model)
        self._entry_row(asr_card, 8, "GigaAM model", self.gigaam_model)
        self._entry_row(asr_card, 9, "GigaAM device", self.gigaam_device)
        self._entry_row(asr_card, 10, "GigaAM segment sec", self.gigaam_segment_sec)
        self._entry_row(asr_card, 11, "Merge similarity threshold", self.merge_similarity_threshold)
        ttk.Checkbutton(asr_card, text="Split stereo channels", variable=self.split_stereo).grid(
            row=11, column=2, sticky="w", padx=6, pady=3
        )

        codex_card = ttk.LabelFrame(
            content,
            text="Stage 2: Codex / LLM Post-processing",
            style="Card.TLabelframe",
            padding=10,
        )
        codex_card.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        self._combo_row(
            codex_card,
            0,
            "Resolve LLM provider",
            self.resolve_llm_provider,
            ["codex_cli", "ollama", "openai", "off"],
        )
        self._combo_row(
            codex_card,
            1,
            "Analyze provider",
            self.analyze_provider,
            ["codex_cli", "ollama", "openai", "mock"],
        )
        self._entry_row(codex_card, 2, "Codex command", self.codex_cli_command)
        self._entry_row(codex_card, 3, "Codex model", self.codex_merge_model)
        self._entry_row(codex_card, 4, "Codex timeout sec", self.codex_cli_timeout_sec)
        self._entry_row(codex_card, 5, "Ollama base URL", self.ollama_base_url)
        self._entry_row(codex_card, 6, "Ollama model", self.ollama_model)
        self._entry_row(codex_card, 7, "Ollama think", self.ollama_think)
        self._entry_row(codex_card, 8, "Ollama temperature", self.ollama_temperature)

        reliability_card = ttk.LabelFrame(content, text="Reliability", style="Card.TLabelframe", padding=10)
        reliability_card.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))
        self._entry_row(
            reliability_card,
            0,
            "Mono role mode",
            self.mono_role_assignment_mode,
        )
        self._entry_row(
            reliability_card,
            1,
            "Mono assign min confidence",
            self.mono_role_assignment_min_confidence,
        )
        self._entry_row(
            reliability_card,
            2,
            "Mono assign LLM threshold",
            self.mono_role_assignment_llm_threshold,
        )
        self._entry_row(reliability_card, 3, "OpenAI role model", self.openai_role_assign_model)
        self._entry_row(reliability_card, 4, "Transcribe max attempts", self.transcribe_max_attempts)
        self._entry_row(reliability_card, 5, "Analyze max attempts", self.analyze_max_attempts)
        self._entry_row(reliability_card, 6, "Sync max attempts", self.sync_max_attempts)
        self._entry_row(reliability_card, 7, "Retry base delay sec", self.retry_base_delay_sec)
        self._entry_row(reliability_card, 8, "Worker poll sec", self.worker_poll_sec)
        self._entry_row(reliability_card, 9, "Worker max idle cycles", self.worker_max_idle_cycles)

        actions_card = ttk.LabelFrame(content, text="Actions", style="Card.TLabelframe", padding=10)
        actions_card.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(0, 6))
        for col in range(4):
            actions_card.columnconfigure(col, weight=1)

        ttk.Label(actions_card, text="Setup", style="App.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", padx=4, pady=(0, 2)
        )
        ttk.Button(actions_card, text="1. Init DB", command=self.on_init_db, style="Primary.TButton").grid(
            row=1, column=0, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(actions_card, text="2. Ingest Calls", command=self.on_ingest, style="Primary.TButton").grid(
            row=1, column=1, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(actions_card, text="3. Refresh Stats", command=self.on_refresh_stats, style="Primary.TButton").grid(
            row=1, column=2, padx=4, pady=4, sticky="ew"
        )

        ttk.Label(actions_card, text="Stage 1: Recognition only", style="App.TLabel").grid(
            row=2, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 2)
        )
        ttk.Button(
            actions_card,
            text="Stage 1: Transcribe Batch",
            command=self.on_transcribe_batch,
            style="Primary.TButton",
        ).grid(row=3, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(
            actions_card,
            text="Stage 1: Transcribe All Pending",
            command=self.on_transcribe_drain,
            style="Primary.TButton",
        ).grid(row=3, column=1, padx=4, pady=4, sticky="ew")

        ttk.Label(actions_card, text="Stage 2: Post-processing", style="App.TLabel").grid(
            row=4, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 2)
        )
        ttk.Button(
            actions_card,
            text="Codex Resolve Batch",
            command=self.on_codex_resolve_batch,
            style="Primary.TButton",
        ).grid(row=5, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(
            actions_card,
            text="Codex Analyze Batch",
            command=self.on_codex_analyze_batch,
            style="Primary.TButton",
        ).grid(row=5, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(
            actions_card,
            text="Codex Post-process All Pending",
            command=self.on_codex_postprocess_drain,
            style="Primary.TButton",
        ).grid(row=5, column=2, padx=4, pady=4, sticky="ew")

        secondary_actions = [
            ("Resolve Batch (current)", self.on_resolve_batch),
            ("Analyze Batch (current)", self.on_analyze_batch),
            ("Sync Batch", self.on_sync_batch),
            ("Worker Once", self.on_worker_once),
            ("Worker Start", self.on_worker_start),
            ("Worker Stop", self.on_worker_stop),
            ("Requeue Dead", self.on_requeue_dead),
            ("Reset Missing Variants", self.on_reset_missing_variants),
            ("Export Review Queue", self.on_export_review_queue),
            ("Export Failed Resolve", self.on_export_failed_resolve_queue),
            ("Export CRM Fields", self.on_export_crm_fields),
        ]
        ttk.Label(actions_card, text="Advanced Operations", style="App.TLabel").grid(
            row=6, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 2)
        )
        base = 7
        for idx, (label, callback) in enumerate(secondary_actions):
            ttk.Button(actions_card, text=label, command=callback).grid(
                row=base + idx // 4,
                column=idx % 4,
                padx=4,
                pady=4,
                sticky="ew",
            )

        ttk.Label(
            actions_card,
            style="Muted.TLabel",
            wraplength=1220,
            justify=tk.LEFT,
            text=(
                "Resume-safe behavior: transcribe/resolve/analyze commands work on pending+failed only. "
                "Rows with done status are preserved and skipped automatically, so no duplicate processing occurs. "
                "Use batch size above to control how many calls are processed per iteration."
            ),
        ).grid(row=base + 3, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 2))

    def _build_status_tab(self, parent: ttk.Frame) -> None:
        content = self._make_scrollable_tab(parent)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(2, weight=1)

        cards = ttk.Frame(content, style="App.TFrame")
        cards.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        for col in range(4):
            cards.columnconfigure(col, weight=1)

        card_specs = [
            ("Total calls", "total_calls"),
            ("Transcribe done", "tr_done"),
            ("Resolve done", "rs_done"),
            ("Analyze done", "an_done"),
            ("Transcribe pending", "tr_pending"),
            ("Resolve pending", "rs_pending"),
            ("Analyze pending", "an_pending"),
            ("Manual review", "rs_manual"),
        ]
        for idx, (caption, key) in enumerate(card_specs):
            card = ttk.LabelFrame(cards, text=caption, style="Card.TLabelframe", padding=10)
            card.grid(row=idx // 4, column=idx % 4, sticky="nsew", padx=4, pady=4)
            ttk.Label(card, textvariable=self.metric_vars[key], style="CardValue.TLabel").pack(anchor="w")
            ttk.Label(card, text=f"metric: {key}", style="CardCaption.TLabel").pack(anchor="w", pady=(4, 0))

        progress = ttk.LabelFrame(content, text="Progress", style="Card.TLabelframe", padding=10)
        progress.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        progress.columnconfigure(1, weight=1)
        ttk.Label(progress, text="Transcribe completion", style="App.TLabel").grid(
            row=0, column=0, sticky="w", padx=4, pady=4
        )
        self.transcribe_progress = ttk.Progressbar(
            progress, style="Thin.Horizontal.TProgressbar", maximum=100, value=0
        )
        self.transcribe_progress.grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(progress, text="Analyze completion", style="App.TLabel").grid(
            row=1, column=0, sticky="w", padx=4, pady=4
        )
        self.analyze_progress = ttk.Progressbar(
            progress, style="Thin.Horizontal.TProgressbar", maximum=100, value=0
        )
        self.analyze_progress.grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(progress, text="Refresh Now", command=self.on_refresh_stats, style="Primary.TButton").grid(
            row=0, column=2, rowspan=2, padx=6, pady=4, sticky="nsew"
        )

        stats_wrap = ttk.LabelFrame(content, text="Raw Stats JSON", style="Card.TLabelframe", padding=8)
        stats_wrap.grid(row=2, column=0, columnspan=2, sticky="nsew")
        stats_wrap.rowconfigure(0, weight=1)
        stats_wrap.columnconfigure(0, weight=1)
        self.stats_text = tk.Text(stats_wrap, wrap=tk.WORD, height=16)
        self.stats_text.grid(row=0, column=0, sticky="nsew")
        stats_scroll = ttk.Scrollbar(stats_wrap, command=self.stats_text.yview)
        stats_scroll.grid(row=0, column=1, sticky="ns")
        self.stats_text.configure(yscrollcommand=stats_scroll.set)

    def _build_functions_tab(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Functional Map", style="Card.TLabelframe", padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        text = tk.Text(frame, wrap=tk.WORD)
        text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, command=text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scroll.set)
        text.insert(tk.END, self._functions_reference_text())
        text.configure(state=tk.DISABLED)

    def _build_design_tab(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(
            parent,
            text="Apple-style Design Review (simulated persona)",
            style="Card.TLabelframe",
            padding=10,
        )
        frame.pack(fill=tk.BOTH, expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        text = tk.Text(frame, wrap=tk.WORD)
        text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, command=text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scroll.set)
        text.insert(tk.END, self._design_review_text())
        text.configure(state=tk.DISABLED)

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Execution Log", style="Card.TLabelframe", padding=8)
        panel.pack(fill=tk.BOTH, expand=True)
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)
        ttk.Label(
            panel,
            style="Muted.TLabel",
            text="Tip: this is append-only runtime log. Use it to monitor long queue drains and retries.",
        ).grid(row=0, column=0, sticky="w", padx=2, pady=(0, 6))
        self.log = tk.Text(panel, wrap=tk.WORD)
        self.log.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(panel, command=self.log.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)
        actions = ttk.Frame(panel, style="App.TFrame")
        actions.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(actions, text="Clear Log", command=self.on_clear_log).pack(side=tk.LEFT)

    def _path_row(
        self,
        frame: ttk.LabelFrame,
        row: int,
        label: str,
        var: tk.StringVar,
        ask_func,
    ) -> None:
        ttk.Label(frame, text=label, style="App.TLabel").grid(row=row, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(frame, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(frame, text="Browse", command=ask_func).grid(
            row=row,
            column=2,
            sticky="ew",
            padx=6,
            pady=3,
        )
        frame.columnconfigure(1, weight=1)

    def _entry_row(self, frame: ttk.LabelFrame, row: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(frame, text=label, style="App.TLabel").grid(row=row, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(frame, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        frame.columnconfigure(1, weight=1)

    def _combo_row(
        self,
        frame: ttk.LabelFrame,
        row: int,
        label: str,
        var: tk.StringVar,
        values: list[str],
    ) -> None:
        ttk.Label(frame, text=label, style="App.TLabel").grid(row=row, column=0, sticky="w", padx=6, pady=3)
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

    def _backend_python_exe(self) -> str:
        candidate = self.backend_python.get().strip()
        if candidate:
            return candidate
        return sys.executable

    def _build_env(self, overrides: Optional[dict[str, str]] = None) -> dict[str, str]:
        env = os.environ.copy()
        project = Path(self.project_dir.get()).expanduser().resolve()
        if self.use_project_src.get():
            env["PYTHONPATH"] = str(project / "src")
        else:
            env.pop("PYTHONPATH", None)
        env["PATH"] = f"{project / '.local' / 'bin'}:{env.get('PATH', '')}"
        env["DATABASE_URL"] = f"sqlite:///{Path(self.database_path.get()).expanduser().resolve()}"
        env["TRANSCRIPT_EXPORT_DIR"] = self.export_dir.get().strip()
        env.update(self._transcribe_mode_env())
        env["DUAL_MERGE_PROVIDER"] = self.merge_provider.get().strip()
        env["RESOLVE_LLM_PROVIDER"] = self.resolve_llm_provider.get().strip()
        env["ANALYZE_PROVIDER"] = self.analyze_provider.get().strip()
        env["DUAL_MERGE_SIMILARITY_THRESHOLD"] = self.merge_similarity_threshold.get().strip()
        env["CODEX_MERGE_MODEL"] = self.codex_merge_model.get().strip()
        env["CODEX_CLI_COMMAND"] = self.codex_cli_command.get().strip()
        env["CODEX_CLI_TIMEOUT_SEC"] = self.codex_cli_timeout_sec.get().strip()
        env["SPLIT_STEREO_CHANNELS"] = "1" if self.split_stereo.get() else "0"
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
        if overrides:
            for key, value in overrides.items():
                env[key] = str(value)
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
        self.after(LOG_POLL_MS, self._drain_log_queue)

    def _set_run_state(self, value: str) -> None:
        self.run_state.set(value)

    def _extract_json_payload(self, text: str) -> dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        best: dict[str, Any] = {}
        for idx, ch in enumerate(raw):
            if ch != "{":
                continue
            try:
                obj, _end = decoder.raw_decode(raw[idx:])
                if isinstance(obj, dict):
                    best = obj
            except json.JSONDecodeError:
                continue
        return best

    def _run_cli(
        self,
        args: list[str],
        env_overrides: Optional[dict[str, str]] = None,
        quiet: bool = False,
    ) -> tuple[int, dict[str, Any]]:
        cmd = [self._backend_python_exe(), "-u", "-m", "mango_mvp.cli", *args]
        env = self._build_env(env_overrides)
        if not quiet:
            self._append_log(f"$ {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            cwd=self.project_dir.get().strip() or str(Path.cwd()),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with self.current_process_lock:
            self.current_process = proc

        try:
            while proc.poll() is None:
                if self.stop_event.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
                time.sleep(0.1)
            stdout, stderr = proc.communicate()
            if stdout and not quiet:
                self._append_log(stdout)
            if stderr and not quiet:
                self._append_log(stderr)
                lowered = stderr.lower()
                if "no module named 'sqlalchemy'" in lowered or 'no module named "sqlalchemy"' in lowered:
                    self._append_log(
                        "[hint] Selected Backend Python does not have project dependencies. "
                        "Choose stable runtime python: stable_runtime/venv_stable/bin/python "
                        "or start UI via ./stable_runtime/run-ui.sh"
                    )
            payload = self._extract_json_payload(stdout)
            if not quiet:
                self._append_log(f"[exit_code={proc.returncode}]")
            return int(proc.returncode or 0), payload
        finally:
            with self.current_process_lock:
                if self.current_process is proc:
                    self.current_process = None

    def _limit_value(self) -> int:
        raw = self.stage_limit.get().strip()
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
        return 100

    def _transcribe_mode_env(self) -> dict[str, str]:
        mode = self.transcribe_mode.get().strip().lower()
        primary = self.transcribe_provider.get().strip() or "mlx"
        secondary = self.secondary_provider.get().strip() or "gigaam"
        if mode == "whisper":
            return {
                "TRANSCRIBE_PROVIDER": "mlx",
                "DUAL_TRANSCRIBE_ENABLED": "0",
                "SECONDARY_TRANSCRIBE_PROVIDER": "",
            }
        if mode == "gigaam":
            return {
                "TRANSCRIBE_PROVIDER": "gigaam",
                "DUAL_TRANSCRIBE_ENABLED": "0",
                "SECONDARY_TRANSCRIBE_PROVIDER": "",
            }
        return {
            "TRANSCRIBE_PROVIDER": primary,
            "DUAL_TRANSCRIBE_ENABLED": "1",
            "SECONDARY_TRANSCRIBE_PROVIDER": secondary,
        }

    def _launch_task(self, title: str, fn) -> None:
        if self.active_task and self.active_task.is_alive():
            self._append_log(f"[busy] another task is running: {self.run_state.get()}")
            return

        self.stop_event.clear()
        self._append_log(f"\n=== {title} ===")
        self._set_run_state(title)

        def _runner() -> None:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                self._append_log(f"[error] {exc}")
            finally:
                self._append_log(f"=== {title}: finished ===")
                self.after(0, self._set_run_state, "Idle")
                self.after(0, self._refresh_stats_async, True)

        self.active_task = threading.Thread(target=_runner, daemon=True)
        self.active_task.start()

    def _format_payload_inline(self, payload: dict[str, Any]) -> str:
        ordered = [
            "processed",
            "success",
            "failed",
            "manual",
            "skipped_short",
            "llm_used",
            "rescue_used",
            "updated",
            "exported",
        ]
        parts: list[str] = []
        for key in ordered:
            if key in payload:
                parts.append(f"{key}={payload.get(key)}")
        return ", ".join(parts)

    def _drain_stage(
        self,
        cmd_name: str,
        *,
        env_overrides: Optional[dict[str, str]] = None,
    ) -> dict[str, int]:
        limit = self._limit_value()
        total_processed = 0
        total_success = 0
        total_failed = 0
        cycle = 0
        while not self.stop_event.is_set():
            cycle += 1
            rc, payload = self._run_cli(
                [cmd_name, "--limit", str(limit)],
                env_overrides=env_overrides,
            )
            if rc != 0:
                self._append_log(f"[{cmd_name}] cycle={cycle} failed with rc={rc}")
                break

            processed = int(payload.get("processed", 0) or 0)
            success = int(payload.get("success", 0) or 0)
            failed = int(payload.get("failed", 0) or 0)
            total_processed += processed
            total_success += success
            total_failed += failed

            self._append_log(
                f"[{cmd_name}] cycle={cycle}: {self._format_payload_inline(payload) or 'no payload'}"
            )
            if processed == 0:
                break
        return {
            "processed": total_processed,
            "success": total_success,
            "failed": total_failed,
        }

    def _refresh_stats_sync(self, silent: bool) -> None:
        rc, payload = self._run_cli(["stats"], quiet=silent)
        if rc != 0 or not payload:
            if not silent:
                self._append_log("[stats] failed to fetch status.")
            return
        self._update_stats_widgets(payload)

    def _refresh_stats_async(self, silent: bool = False) -> None:
        def _runner() -> None:
            self._refresh_stats_sync(silent=silent)

        threading.Thread(target=_runner, daemon=True).start()

    def _schedule_auto_stats(self) -> None:
        if self._auto_stats_job is not None:
            self.after_cancel(self._auto_stats_job)
        self._auto_stats_job = self.after(STATUS_REFRESH_SEC * 1000, self._on_auto_stats_tick)

    def _on_auto_stats_tick(self) -> None:
        self._refresh_stats_async(silent=True)
        self._schedule_auto_stats()

    def _update_stats_widgets(self, payload: dict[str, Any]) -> None:
        total = int(payload.get("total_calls", 0) or 0)
        tr = payload.get("transcription_status") or {}
        rs = payload.get("resolve_status") or {}
        an = payload.get("analysis_status") or {}

        self.metric_vars["total_calls"].set(str(total))
        self.metric_vars["tr_done"].set(str(int(tr.get("done", 0) or 0)))
        self.metric_vars["tr_pending"].set(str(int(tr.get("pending", 0) or 0)))
        self.metric_vars["tr_failed"].set(str(int(tr.get("failed", 0) or 0)))
        self.metric_vars["tr_dead"].set(str(int(tr.get("dead", 0) or 0)))

        self.metric_vars["rs_done"].set(str(int(rs.get("done", 0) or 0)))
        self.metric_vars["rs_skipped"].set(str(int(rs.get("skipped", 0) or 0)))
        self.metric_vars["rs_manual"].set(str(int(rs.get("manual", 0) or 0)))
        self.metric_vars["rs_pending"].set(str(int(rs.get("pending", 0) or 0)))

        self.metric_vars["an_done"].set(str(int(an.get("done", 0) or 0)))
        self.metric_vars["an_pending"].set(str(int(an.get("pending", 0) or 0)))
        self.metric_vars["an_failed"].set(str(int(an.get("failed", 0) or 0)))
        self.metric_vars["an_dead"].set(str(int(an.get("dead", 0) or 0)))

        tr_done = int(tr.get("done", 0) or 0)
        an_done = int(an.get("done", 0) or 0)
        tr_progress = (100.0 * tr_done / total) if total > 0 else 0.0
        an_progress = (100.0 * an_done / total) if total > 0 else 0.0
        self.transcribe_progress["value"] = tr_progress
        self.analyze_progress["value"] = an_progress

        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        self.stats_text.configure(state=tk.NORMAL)
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, rendered)
        self.stats_text.configure(state=tk.DISABLED)

    def on_init_db(self) -> None:
        self._launch_task("Init DB", lambda: self._run_cli(["init-db"]))

    def on_ingest(self) -> None:
        def _task() -> None:
            args = ["ingest", "--recordings-dir", self.recordings_dir.get().strip()]
            meta = self.metadata_csv.get().strip()
            if meta:
                args.extend(["--metadata-csv", meta])
            self._run_cli(args)
            self._refresh_stats_sync(silent=True)

        self._launch_task("Ingest Calls", _task)

    def on_refresh_stats(self) -> None:
        self._launch_task("Refresh Stats", lambda: self._refresh_stats_sync(silent=False))

    def on_transcribe_batch(self) -> None:
        def _task() -> None:
            env = self._transcribe_mode_env()
            self._run_cli(["transcribe", "--limit", str(self._limit_value())], env_overrides=env)
            self._refresh_stats_sync(silent=True)

        self._launch_task("Stage 1: Transcribe Batch", _task)

    def on_transcribe_drain(self) -> None:
        def _task() -> None:
            env = self._transcribe_mode_env()
            summary = self._drain_stage("transcribe", env_overrides=env)
            self._append_log(f"[transcribe drain] summary: {summary}")
            self._refresh_stats_sync(silent=True)

        self._launch_task("Stage 1: Transcribe All Pending", _task)

    def on_codex_resolve_batch(self) -> None:
        def _task() -> None:
            env = {"RESOLVE_LLM_PROVIDER": "codex_cli"}
            self._run_cli(["resolve", "--limit", str(self._limit_value())], env_overrides=env)
            self._refresh_stats_sync(silent=True)

        self._launch_task("Codex Resolve Batch", _task)

    def on_codex_analyze_batch(self) -> None:
        def _task() -> None:
            env = {"ANALYZE_PROVIDER": "codex_cli"}
            self._run_cli(["analyze", "--limit", str(self._limit_value())], env_overrides=env)
            self._refresh_stats_sync(silent=True)

        self._launch_task("Codex Analyze Batch", _task)

    def on_codex_postprocess_drain(self) -> None:
        def _task() -> None:
            resolve_summary = self._drain_stage(
                "resolve",
                env_overrides={"RESOLVE_LLM_PROVIDER": "codex_cli"},
            )
            self._append_log(f"[codex resolve drain] summary: {resolve_summary}")
            if self.stop_event.is_set():
                return
            analyze_summary = self._drain_stage(
                "analyze",
                env_overrides={"ANALYZE_PROVIDER": "codex_cli"},
            )
            self._append_log(f"[codex analyze drain] summary: {analyze_summary}")
            self._refresh_stats_sync(silent=True)

        self._launch_task("Codex Post-process All Pending", _task)

    def on_resolve_batch(self) -> None:
        self._launch_task(
            "Resolve Batch (current provider)",
            lambda: self._run_cli(["resolve", "--limit", str(self._limit_value())]),
        )

    def on_analyze_batch(self) -> None:
        self._launch_task(
            "Analyze Batch (current provider)",
            lambda: self._run_cli(["analyze", "--limit", str(self._limit_value())]),
        )

    def on_sync_batch(self) -> None:
        self._launch_task(
            "Sync Batch",
            lambda: self._run_cli(["sync", "--limit", str(self._limit_value())]),
        )

    def on_worker_once(self) -> None:
        self._launch_task(
            "Worker Once",
            lambda: self._run_cli(["worker", "--once", "--stage-limit", str(self._limit_value())]),
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
            str(self._limit_value()),
            "--poll-sec",
            self.worker_poll_sec.get().strip() or "10",
            "--max-idle-cycles",
            self.worker_max_idle_cycles.get().strip() or "30",
        ]
        self._append_log("\n=== Worker Start ===")
        self._append_log(f"$ {' '.join(cmd)}")
        try:
            self.worker_process = subprocess.Popen(
                cmd,
                cwd=self.project_dir.get().strip() or str(Path.cwd()),
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

    def on_requeue_dead(self) -> None:
        self._launch_task("Requeue Dead", lambda: self._run_cli(["requeue-dead", "--stage", "all"]))

    def on_reset_missing_variants(self) -> None:
        self._launch_task(
            "Reset Missing Variants",
            lambda: self._run_cli(
                [
                    "reset-transcribe",
                    "--only-done",
                    "--only-missing-variants",
                    "--limit",
                    str(self._limit_value()),
                ]
            ),
        )

    def on_export_review_queue(self) -> None:
        out_path = Path(self.project_dir.get()) / "manual_review_queue.csv"
        self._launch_task(
            "Export Manual Review Queue",
            lambda: self._run_cli(
                ["export-review-queue", "--out", str(out_path), "--limit", "100000"]
            ),
        )

    def on_export_failed_resolve_queue(self) -> None:
        out_path = Path(self.project_dir.get()) / "failed_resolve_queue.csv"
        self._launch_task(
            "Export Failed Resolve Queue",
            lambda: self._run_cli(
                ["export-failed-resolve-queue", "--out", str(out_path), "--limit", "100000"]
            ),
        )

    def on_export_crm_fields(self) -> None:
        out_path = Path(self.project_dir.get()) / "crm_fields.csv"
        self._launch_task(
            "Export CRM Fields",
            lambda: self._run_cli(
                [
                    "export-crm-fields",
                    "--out",
                    str(out_path),
                    "--only-done",
                    "--limit",
                    "200000",
                ]
            ),
        )

    def on_stop_current_task(self) -> None:
        self.stop_event.set()
        with self.current_process_lock:
            proc = self.current_process
        if proc and proc.poll() is None:
            proc.terminate()
            self._append_log("[task] terminate requested for current CLI process.")
        else:
            self._append_log("[task] no active CLI process.")

    def on_clear_log(self) -> None:
        self.log.delete("1.0", tk.END)

    def _functions_reference_text(self) -> str:
        return (
            "Mango Calls Studio - function reference\n\n"
            "Recommended flow\n"
            "1. Init DB\n"
            "   Creates DB schema in the selected SQLite file.\n\n"
            "2. Ingest Calls\n"
            "   Scans recordings directory, parses Mango filename metadata, and inserts only new files.\n"
            "   Already indexed source_file rows are skipped automatically.\n\n"
            "3. Stage 1: Recognition only\n"
            "   Use Transcribe mode + Records per batch field, then click:\n"
            "   - Stage 1: Transcribe Batch\n"
            "   - Stage 1: Transcribe All Pending\n"
            "   Modes:\n"
            "   - whisper: MLX Whisper only\n"
            "   - gigaam: GigaAM only\n"
            "   - dual: Whisper + GigaAM merge\n"
            "   Resume-safe behavior: command processes pending+failed only; done rows are preserved.\n\n"
            "4. Stage 2: Codex post-processing\n"
            "   Post-processing pipeline with Codex CLI:\n"
            "   - Resolve improves risky transcripts and quality flags\n"
            "   - Analyze builds CRM-ready summaries/structured fields\n"
            "   Post-process commands are independent from transcription stage.\n\n"
            "5. Codex Post-process All Pending\n"
            "   Drains resolve queue, then drains analyze queue until processed=0.\n\n"
            "Operational and recovery tools\n"
            "- Worker Start/Stop: continuous resilient loop.\n"
            "- Requeue Dead: moves dead-letter rows back to pending.\n"
            "- Reset Missing Variants: re-opens done transcripts that miss dual-ASR variants.\n"
            "- Export Review Queue: CSV for manual QA.\n"
            "- Export Failed Resolve: CSV with failed/dead resolve calls.\n"
            "- Export CRM Fields: flatten analyzed payload into CSV columns.\n\n"
            "Idempotency guarantee\n"
            "- Pipeline stages use status columns in DB.\n"
            "- Done rows are not reprocessed by normal commands.\n"
            "- Double work happens only if you explicitly reset statuses.\n\n"
            "Where results are stored\n"
            "- DB statuses and payloads are kept in selected SQLite database.\n"
            "- Transcripts and analysis files are exported into selected transcript directory.\n"
        )

    def _design_review_text(self) -> str:
        return (
            "Simulated Apple-style reviewer notes (not a real Apple employee)\n\n"
            "Iteration 1 - clarity\n"
            "- Problem: old UI mixed too many controls in one long panel.\n"
            "- Change: introduced tabbed layout (Pipeline / Status / Functions / Design Review).\n"
            "- Result: user can execute pipeline without scrolling through technical noise.\n\n"
            "Iteration 2 - operational confidence\n"
            "- Problem: it was unclear if completed calls can be reprocessed accidentally.\n"
            "- Change: explicit resume-safe messaging, status cards, progress bars, and queue drains.\n"
            "- Result: operator sees pending/done/failed and can continue from exact checkpoint.\n\n"
            "Iteration 3 - stage ergonomics\n"
            "- Problem: stage 1 (ASR) and stage 2 (Codex) were visually mixed.\n"
            "- Change: explicit Stage 1/Stage 2 action blocks and clearer naming.\n"
            "- Result: users can run recognition-only flow with minimal cognitive load.\n\n"
            "Iteration 4 - accessibility/scroll\n"
            "- Problem: some controls were clipped on smaller windows.\n"
            "- Change: added scrollable tab containers with mouse wheel support.\n"
            "- Result: all controls are reachable without resizing the app.\n\n"
            "Iteration 5 - dual mode confusion\n"
            "- Problem: dual checkbox with cross/check state looked ambiguous.\n"
            "- Change: removed ambiguous toggle; transcribe mode now controls behavior directly.\n"
            "- Result: no conflicting state, one obvious source of truth.\n\n"
            "Reviewer verdict\n"
            "- Information hierarchy is now clear.\n"
            "- Primary actions are front-loaded.\n"
            "- Status visibility supports long-running operations.\n"
            "- Current UI is stable for semi-production operators.\n"
        )


def main() -> int:
    app = MangoMvpGui()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
