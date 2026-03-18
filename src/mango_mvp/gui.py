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
        self._scroll_canvases: list[tk.Canvas] = []
        self._advanced_widgets: list[tk.Widget] = []

        self.project_dir = tk.StringVar(value=str(Path.cwd()))
        self.recordings_dir = tk.StringVar(value=str(Path.cwd()))
        self.metadata_csv = tk.StringVar(value="")
        self.database_path = tk.StringVar(value=str(Path.cwd() / "mango_mvp.db"))
        self.export_dir = tk.StringVar(value=str(Path.cwd() / "transcripts"))
        self.simple_mode = tk.BooleanVar(value=True)

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
        self.resolve_dialogue_mode = tk.StringVar(value="dialogue")
        self.analyze_provider = tk.StringVar(value="codex_cli")
        self.codex_merge_model = tk.StringVar(value="gpt-5.4")
        self.codex_cli_command = tk.StringVar(value="codex")
        self.codex_cli_timeout_sec = tk.StringVar(value="180")
        self.codex_reasoning_effort = tk.StringVar(value="medium")
        self.pilot_seed = tk.StringVar(value="42")
        self.pilot_ids_path = tk.StringVar(
            value=str(Path.cwd() / "stable_runtime" / "pilots" / "resolve_pilot_ids.txt")
        )
        self.pilot_export_dir = tk.StringVar(
            value=str(Path.cwd() / "pilot_exports" / "resolve_pilot")
        )

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
        self.pipeline_stage_transcribe = tk.BooleanVar(value=True)
        self.pipeline_stage_backfill = tk.BooleanVar(value=True)
        self.pipeline_stage_resolve = tk.BooleanVar(value=True)
        self.pipeline_stage_analyze = tk.BooleanVar(value=True)
        self.pipeline_stage_sync = tk.BooleanVar(value=False)

        self.run_state = tk.StringVar(value="Ожидание")
        self.batch_progress = tk.StringVar(value="Батч: —")
        self.secondary_asr_summary = tk.StringVar(value="Очередь 2-го ASR: —")
        self.metric_vars = {
            "total_calls": tk.StringVar(value="0"),
            "secondary_asr_pending": tk.StringVar(value="0"),
            "tr_done": tk.StringVar(value="0"),
            "tr_pending": tk.StringVar(value="0"),
            "tr_failed": tk.StringVar(value="0"),
            "tr_dead": tk.StringVar(value="0"),
            "rs_done": tk.StringVar(value="0"),
            "rs_skipped": tk.StringVar(value="0"),
            "rs_manual": tk.StringVar(value="0"),
            "rs_pending": tk.StringVar(value="0"),
            "rs_failed": tk.StringVar(value="0"),
            "rs_dead": tk.StringVar(value="0"),
            "an_done": tk.StringVar(value="0"),
            "an_pending": tk.StringVar(value="0"),
            "an_failed": tk.StringVar(value="0"),
            "an_dead": tk.StringVar(value="0"),
            "sy_done": tk.StringVar(value="0"),
            "sy_pending": tk.StringVar(value="0"),
            "sy_failed": tk.StringVar(value="0"),
            "sy_dead": tk.StringVar(value="0"),
            "dl_transcribe": tk.StringVar(value="0"),
            "dl_resolve": tk.StringVar(value="0"),
            "dl_analyze": tk.StringVar(value="0"),
            "dl_sync": tk.StringVar(value="0"),
        }

        self._setup_theme()
        self.transcribe_mode.trace_add("write", self._on_transcribe_mode_changed)
        self._on_transcribe_mode_changed()
        self._build_ui()
        self.bind_all("<MouseWheel>", self._on_global_mousewheel, add="+")
        self.bind_all("<Button-4>", self._on_global_mousewheel, add="+")
        self.bind_all("<Button-5>", self._on_global_mousewheel, add="+")
        self.simple_mode.trace_add("write", self._on_simple_mode_changed)
        self.after(0, self._apply_simple_mode)
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
                "Стабильная локальная панель: очередь распознавания, очередь постобработки Codex и "
                "безопасные перезапуски (звонки со статусом done не обрабатываются повторно, пока вы "
                "вручную не сбросите статусы)."
            ),
            style="SubHeader.TLabel",
            wraplength=1180,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(2, 6))

        status_row = ttk.Frame(header, style="App.TFrame")
        status_row.pack(fill=tk.X)
        ttk.Label(status_row, text="Состояние:", style="Muted.TLabel").pack(side=tk.LEFT)
        ttk.Label(status_row, textvariable=self.run_state, style="App.TLabel").pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(status_row, textvariable=self.batch_progress, style="Muted.TLabel").pack(
            side=tk.LEFT, padx=(16, 0)
        )
        ttk.Checkbutton(
            status_row,
            text="Простой режим",
            variable=self.simple_mode,
            command=self._apply_simple_mode,
        ).pack(side=tk.RIGHT, padx=(0, 10))
        ttk.Button(
            status_row,
            text="Остановить текущую задачу",
            command=self.on_stop_current_task,
            style="Primary.TButton",
        ).pack(side=tk.RIGHT)

        top = ttk.Panedwindow(root, orient=tk.VERTICAL)
        top.pack(fill=tk.BOTH, expand=True)

        notebook_wrap = ttk.Frame(top, style="App.TFrame")
        logs_wrap = ttk.Frame(top, style="App.TFrame")
        top.add(notebook_wrap, weight=4)
        top.add(logs_wrap, weight=2)

        self.notebook = ttk.Notebook(notebook_wrap)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_pipeline = ttk.Frame(self.notebook, style="App.TFrame")
        self.tab_status = ttk.Frame(self.notebook, style="App.TFrame")
        self.tab_functions = ttk.Frame(self.notebook, style="App.TFrame")
        self.tab_design = ttk.Frame(self.notebook, style="App.TFrame")
        self.notebook.add(self.tab_pipeline, text="Конвейер")
        self.notebook.add(self.tab_status, text="Статус")
        self.notebook.add(self.tab_functions, text="Справка")
        self.notebook.add(self.tab_design, text="Дизайн-ревью")

        self._build_pipeline_tab(self.tab_pipeline)
        self._build_status_tab(self.tab_status)
        self._build_functions_tab(self.tab_functions)
        self._build_design_tab(self.tab_design)
        self._build_log_panel(logs_wrap)

    def _on_transcribe_mode_changed(self, *_: object) -> None:
        mode = self.transcribe_mode.get().strip().lower()
        if mode == "dual":
            self.dual_mode_state.set("Dual: ВКЛ (две ASR-системы)")
            if not self.secondary_provider.get().strip():
                self.secondary_provider.set("gigaam")
        elif mode == "whisper":
            self.dual_mode_state.set("Dual: ВЫКЛ (только Whisper)")
        elif mode == "gigaam":
            self.dual_mode_state.set("Dual: ВЫКЛ (только GigaAM)")
        else:
            self.dual_mode_state.set("Dual: ВЫКЛ")

    def _on_simple_mode_changed(self, *_: object) -> None:
        self._apply_simple_mode()

    def _apply_simple_mode(self) -> None:
        simple = bool(self.simple_mode.get())

        if hasattr(self, "codex_card"):
            if simple:
                self.codex_card.grid_remove()
            else:
                self.codex_card.grid()

        if hasattr(self, "reliability_card"):
            if simple:
                self.reliability_card.grid_remove()
            else:
                self.reliability_card.grid()

        for widget in getattr(self, "_advanced_widgets", []):
            if simple:
                widget.grid_remove()
            else:
                widget.grid()

        if hasattr(self, "notebook"):
            self._set_notebook_tab_visible(self.tab_functions, text="Справка", visible=not simple)
            self._set_notebook_tab_visible(self.tab_design, text="Дизайн-ревью", visible=not simple)

    def _set_notebook_tab_visible(self, frame: ttk.Frame, *, text: str, visible: bool) -> None:
        if not hasattr(self, "notebook"):
            return
        try:
            state = str(self.notebook.tab(frame, "state"))
        except tk.TclError:
            return

        if visible:
            if state == "hidden":
                self.notebook.add(frame)
            self.notebook.tab(frame, text=text)
        else:
            if state != "hidden":
                self.notebook.hide(frame)

    def _find_scroll_canvas_for_widget(self, widget: Optional[tk.Widget]) -> Optional[tk.Canvas]:
        cur = widget
        while cur is not None:
            if isinstance(cur, tk.Canvas) and cur in self._scroll_canvases:
                return cur
            parent_name = cur.winfo_parent()
            if not parent_name:
                break
            try:
                cur = self.nametowidget(parent_name)
            except Exception:  # noqa: BLE001
                break
        return None

    @staticmethod
    def _mousewheel_step(event: tk.Event) -> int:
        step = 0
        num = getattr(event, "num", None)
        if num == 4:
            return -1
        if num == 5:
            return 1

        delta = int(getattr(event, "delta", 0) or 0)
        if delta == 0:
            return 0
        if abs(delta) >= 120:
            step = int(-delta / 120)
        else:
            step = -1 if delta > 0 else 1
        if step == 0:
            step = -1 if delta > 0 else 1
        return step

    def _scroll_target_widget(self, widget: tk.Widget, event: tk.Event) -> Optional[str]:
        step = self._mousewheel_step(event)
        if step == 0:
            return None
        try:
            widget.yview_scroll(step, "units")
        except Exception:  # noqa: BLE001
            return None
        return "break"

    def _bind_scroll_target(self, widget: tk.Widget, target: tk.Widget) -> None:
        bound_key = f"{widget}.{target}"
        if getattr(widget, "_mango_scroll_bound_key", "") == bound_key:
            return
        widget._mango_scroll_bound_key = bound_key  # type: ignore[attr-defined]
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(
                sequence,
                lambda event, scroll_target=target: self._scroll_target_widget(
                    scroll_target, event
                ),
                add="+",
            )

    def _bind_scroll_descendants(self, widget: tk.Widget, canvas: tk.Canvas) -> None:
        if isinstance(widget, (tk.Text, tk.Listbox)):
            return
        self._bind_scroll_target(widget, canvas)
        for child in widget.winfo_children():
            self._bind_scroll_descendants(child, canvas)

    def _on_global_mousewheel(self, event: tk.Event) -> Optional[str]:
        try:
            pointer_x, pointer_y = self.winfo_pointerxy()
            pointer_widget = self.winfo_containing(pointer_x, pointer_y)
        except Exception:  # noqa: BLE001
            return None

        if pointer_widget is None:
            return None

        # Let native text widgets keep their own smooth scroll behavior.
        if isinstance(pointer_widget, (tk.Text, tk.Listbox)):
            return None

        canvas = self._find_scroll_canvas_for_widget(pointer_widget)
        if canvas is None:
            return None

        return self._scroll_target_widget(canvas, event)

    def _make_scrollable_tab(self, parent: ttk.Frame) -> ttk.Frame:
        wrap = ttk.Frame(parent, style="App.TFrame")
        wrap.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(wrap, bg="#F2F4F7", highlightthickness=0, borderwidth=0)
        self._scroll_canvases.append(canvas)
        vbar = ttk.Scrollbar(wrap, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)

        content = ttk.Frame(canvas, style="App.TFrame")
        win_id = canvas.create_window((0, 0), window=content, anchor="nw")

        def _sync_scroll_region(_event: object) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))
            self._bind_scroll_descendants(content, canvas)

        def _sync_width(_event: object) -> None:
            canvas.itemconfigure(win_id, width=canvas.winfo_width())

        def _on_destroy(_event: object) -> None:
            if canvas in self._scroll_canvases:
                self._scroll_canvases.remove(canvas)

        content.bind("<Configure>", _sync_scroll_region)
        canvas.bind("<Configure>", _sync_width)
        wrap.bind("<Destroy>", _on_destroy)
        self._bind_scroll_target(canvas, canvas)
        self._bind_scroll_descendants(content, canvas)
        return content

    def _build_pipeline_tab(self, parent: ttk.Frame) -> None:
        content = self._make_scrollable_tab(parent)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)

        paths_card = ttk.LabelFrame(content, text="Проект и пути", style="Card.TLabelframe", padding=10)
        paths_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        self._path_row(paths_card, 0, "Папка проекта", self.project_dir, self._pick_project_dir)
        self._path_row(paths_card, 1, "Папка с записями", self.recordings_dir, self._pick_recordings_dir)
        self._path_row(paths_card, 2, "Metadata CSV (опционально)", self.metadata_csv, self._pick_metadata_csv)
        self._path_row(paths_card, 3, "Файл БД", self.database_path, self._pick_database_file)
        self._path_row(paths_card, 4, "Папка экспорта транскриптов", self.export_dir, self._pick_export_dir)
        self._path_row(paths_card, 5, "Backend Python", self.backend_python, self._pick_backend_python)
        ttk.Checkbutton(
            paths_card,
            text="Использовать project src (dev-режим). Для stable snapshot выключайте.",
            variable=self.use_project_src,
        ).grid(row=6, column=1, sticky="w", padx=6, pady=4)

        asr_card = ttk.LabelFrame(
            content, text="Этап 1: Настройка распознавания", style="Card.TLabelframe", padding=10
        )
        asr_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))

        self._combo_row(
            asr_card,
            0,
            "Режим распознавания",
            self.transcribe_mode,
            ["dual", "whisper", "gigaam"],
        )
        self._combo_row(
            asr_card,
            1,
            "Основной провайдер",
            self.transcribe_provider,
            ["mlx", "gigaam", "openai", "mock"],
        )
        self._combo_row(
            asr_card,
            2,
            "Резервный провайдер",
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
            "Провайдер склейки dual",
            self.merge_provider,
            ["rule", "codex_cli", "ollama", "openai", "primary"],
        )
        ttk.Label(asr_card, text="Записей за батч (--limit)", style="App.TLabel").grid(
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
        ttk.Label(preset, text="Быстро:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 4))
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
                "Чтобы запустить только распознавание: выберите режим и нажмите "
                "'Этап 1: Распознать все pending'. Количество за проход задается этим полем."
            ),
            style="Muted.TLabel",
            wraplength=440,
            justify=tk.LEFT,
        ).grid(row=5, column=0, columnspan=3, sticky="w", padx=6, pady=(2, 6))

        self._entry_row(asr_card, 6, "Язык", self.language)
        self._entry_row(asr_card, 7, "Модель MLX Whisper", self.mlx_model)
        self._entry_row(asr_card, 8, "Модель GigaAM", self.gigaam_model)
        self._entry_row(asr_card, 9, "Устройство GigaAM", self.gigaam_device)
        self._entry_row(asr_card, 10, "Сегмент GigaAM (сек)", self.gigaam_segment_sec)
        self._entry_row(asr_card, 11, "Порог похожести склейки", self.merge_similarity_threshold)
        ttk.Checkbutton(asr_card, text="Разделять стерео-каналы", variable=self.split_stereo).grid(
            row=11, column=2, sticky="w", padx=6, pady=3
        )

        codex_card = ttk.LabelFrame(
            content,
            text="Этап 2: Постобработка Codex / LLM",
            style="Card.TLabelframe",
            padding=10,
        )
        self.codex_card = codex_card
        codex_card.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        self._combo_row(
            codex_card,
            0,
            "Провайдер LLM для Resolve",
            self.resolve_llm_provider,
            ["codex_cli", "ollama", "openai", "off"],
        )
        self._combo_row(
            codex_card,
            1,
            "Режим Resolve",
            self.resolve_dialogue_mode,
            ["dialogue", "legacy"],
        )
        self._combo_row(
            codex_card,
            2,
            "Провайдер для Analyze",
            self.analyze_provider,
            ["codex_cli", "ollama", "openai", "mock"],
        )
        self._entry_row(codex_card, 3, "Команда Codex CLI", self.codex_cli_command)
        self._entry_row(codex_card, 4, "Модель Codex", self.codex_merge_model)
        self._entry_row(codex_card, 5, "Таймаут Codex (сек)", self.codex_cli_timeout_sec)
        self._combo_row(
            codex_card,
            6,
            "Codex reasoning",
            self.codex_reasoning_effort,
            ["low", "medium", "high"],
        )
        self._entry_row(codex_card, 7, "Ollama base URL", self.ollama_base_url)
        self._entry_row(codex_card, 8, "Ollama model", self.ollama_model)
        self._entry_row(codex_card, 9, "Ollama think", self.ollama_think)
        self._entry_row(codex_card, 10, "Ollama temperature", self.ollama_temperature)
        self._entry_row(codex_card, 11, "Pilot seed", self.pilot_seed)
        self._entry_row(codex_card, 12, "Pilot ids file", self.pilot_ids_path)
        self._entry_row(codex_card, 13, "Pilot export dir", self.pilot_export_dir)

        reliability_card = ttk.LabelFrame(content, text="Надежность", style="Card.TLabelframe", padding=10)
        self.reliability_card = reliability_card
        reliability_card.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))
        self._entry_row(
            reliability_card,
            0,
            "Режим ролей для моно",
            self.mono_role_assignment_mode,
        )
        self._entry_row(
            reliability_card,
            1,
            "Мин. уверенность ролей (моно)",
            self.mono_role_assignment_min_confidence,
        )
        self._entry_row(
            reliability_card,
            2,
            "Порог LLM для ролей (моно)",
            self.mono_role_assignment_llm_threshold,
        )
        self._entry_row(reliability_card, 3, "Модель OpenAI для ролей", self.openai_role_assign_model)
        self._entry_row(reliability_card, 4, "Макс. попыток transcribe", self.transcribe_max_attempts)
        self._entry_row(reliability_card, 5, "Макс. попыток analyze", self.analyze_max_attempts)
        self._entry_row(reliability_card, 6, "Макс. попыток sync", self.sync_max_attempts)
        self._entry_row(reliability_card, 7, "Базовая задержка retry (сек)", self.retry_base_delay_sec)
        self._entry_row(reliability_card, 8, "Пауза опроса worker (сек)", self.worker_poll_sec)
        self._entry_row(reliability_card, 9, "Макс. idle циклов worker", self.worker_max_idle_cycles)

        actions_card = ttk.LabelFrame(content, text="Действия", style="Card.TLabelframe", padding=10)
        actions_card.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(0, 6))
        for col in range(4):
            actions_card.columnconfigure(col, weight=1)

        ttk.Label(actions_card, text="Подготовка", style="App.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", padx=4, pady=(0, 2)
        )
        ttk.Button(
            actions_card, text="1. Инициализировать БД", command=self.on_init_db, style="Primary.TButton"
        ).grid(
            row=1, column=0, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(actions_card, text="2. Загрузить звонки", command=self.on_ingest, style="Primary.TButton").grid(
            row=1, column=1, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(
            actions_card, text="3. Обновить статус", command=self.on_refresh_stats, style="Primary.TButton"
        ).grid(
            row=1, column=2, padx=4, pady=4, sticky="ew"
        )

        ttk.Label(actions_card, text="Этап 1: Только распознавание", style="App.TLabel").grid(
            row=2, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 2)
        )
        ttk.Button(
            actions_card,
            text="Этап 1: Батч распознавания",
            command=self.on_transcribe_batch,
            style="Primary.TButton",
        ).grid(row=3, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(
            actions_card,
            text="Этап 1: Распознать все pending",
            command=self.on_transcribe_drain,
            style="Primary.TButton",
        ).grid(row=3, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(
            actions_card,
            text="Этап 1: Дораспознать 2-й ASR",
            command=self.on_fill_missing_second_asr,
            style="Primary.TButton",
        ).grid(row=3, column=2, padx=4, pady=4, sticky="ew")

        ttk.Label(actions_card, text="Этап 2: Post-ASR ручные операции", style="App.TLabel").grid(
            row=4, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 2)
        )
        ttk.Button(
            actions_card,
            text="Codex Resolve (батч)",
            command=self.on_codex_resolve_batch,
            style="Primary.TButton",
        ).grid(row=5, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(
            actions_card,
            text="Codex Analyze (батч)",
            command=self.on_codex_analyze_batch,
            style="Primary.TButton",
        ).grid(row=5, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(
            actions_card,
            text="Codex: Resolve + Analyze все pending",
            command=self.on_codex_postprocess_drain,
            style="Primary.TButton",
        ).grid(row=5, column=2, padx=4, pady=4, sticky="ew")

        ttk.Label(actions_card, text="Пилот Resolve", style="App.TLabel").grid(
            row=6, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 2)
        )
        ttk.Button(
            actions_card,
            text="Пилот: подготовить выборку",
            command=self.on_prepare_resolve_pilot,
            style="Primary.TButton",
        ).grid(row=7, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(
            actions_card,
            text="Пилот: экспорт initial",
            command=self.on_export_resolve_pilot_initial,
            style="Primary.TButton",
        ).grid(row=7, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(
            actions_card,
            text="Пилот: экспорт final",
            command=self.on_export_resolve_pilot_final,
            style="Primary.TButton",
        ).grid(row=7, column=2, padx=4, pady=4, sticky="ew")
        ttk.Button(
            actions_card,
            text="Пилот: подготовить + initial",
            command=self.on_prepare_and_export_resolve_pilot_initial,
            style="Primary.TButton",
        ).grid(row=7, column=3, padx=4, pady=4, sticky="ew")

        ttk.Label(actions_card, text="Конвейерные этапы", style="App.TLabel").grid(
            row=8, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 2)
        )
        pipeline_stage_row = ttk.Frame(actions_card, style="App.TFrame")
        pipeline_stage_row.grid(row=9, column=0, columnspan=4, sticky="ew", padx=4, pady=(0, 4))
        for idx, (label, var) in enumerate(
            [
                ("1. Whisper", self.pipeline_stage_transcribe),
                ("2. GigaAM", self.pipeline_stage_backfill),
                ("3. Resolve", self.pipeline_stage_resolve),
                ("4. Analyze", self.pipeline_stage_analyze),
                ("5. Sync", self.pipeline_stage_sync),
            ]
        ):
            ttk.Checkbutton(
                pipeline_stage_row,
                text=label,
                variable=var,
            ).grid(row=0, column=idx, sticky="w", padx=(0, 10))
        ttk.Label(
            actions_card,
            text=(
                "Этот блок управляет только кнопкой 'Конвейер: обработать выбранные этапы' "
                "и кнопками worker ниже. Можно, например, отключить Analyze и остановиться на Resolve. "
                "Отдельные кнопки Codex выше всегда выполняют именно свои явные действия."
            ),
            style="Muted.TLabel",
            wraplength=1220,
            justify=tk.LEFT,
        ).grid(row=10, column=0, columnspan=4, sticky="w", padx=4, pady=(0, 4))
        ttk.Button(
            actions_card,
            text="Конвейер: обработать выбранные этапы",
            command=self.on_pipeline_drain,
            style="Primary.TButton",
        ).grid(row=11, column=0, columnspan=2, padx=4, pady=4, sticky="ew")

        secondary_actions = [
            ("Resolve батч (текущий)", self.on_resolve_batch),
            ("Analyze батч (текущий)", self.on_analyze_batch),
            ("Sync батч", self.on_sync_batch),
            ("Worker один цикл (этапы)", self.on_worker_once),
            ("Worker старт (этапы)", self.on_worker_start),
            ("Worker стоп", self.on_worker_stop),
            ("Вернуть dead в очередь", self.on_requeue_dead),
            ("Сбросить missing variants", self.on_reset_missing_variants),
            ("Экспорт очереди проверки", self.on_export_review_queue),
            ("Экспорт failed resolve", self.on_export_failed_resolve_queue),
            ("Экспорт полей CRM", self.on_export_crm_fields),
        ]
        self._advanced_widgets = []
        advanced_label = ttk.Label(actions_card, text="Дополнительные операции", style="App.TLabel")
        advanced_label.grid(
            row=12, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 2)
        )
        self._advanced_widgets.append(advanced_label)
        base = 13
        for idx, (label, callback) in enumerate(secondary_actions):
            btn = ttk.Button(actions_card, text=label, command=callback)
            btn.grid(
                row=base + idx // 4,
                column=idx % 4,
                padx=4,
                pady=4,
                sticky="ew",
            )
            self._advanced_widgets.append(btn)

        ttk.Label(
            actions_card,
            style="Muted.TLabel",
            wraplength=1220,
            justify=tk.LEFT,
            text=(
                "Режим безопасного продолжения: transcribe/resolve/analyze обрабатывают только pending+failed. "
                "Строки со статусом done сохраняются и автоматически пропускаются, поэтому двойной обработки нет. "
                "Размер батча выше управляет количеством звонков за итерацию."
            ),
        ).grid(row=base + 3, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 2))

    def _build_status_tab(self, parent: ttk.Frame) -> None:
        content = self._make_scrollable_tab(parent)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(4, weight=1)

        explainer = ttk.LabelFrame(
            content,
            text="Как читать статус",
            style="Card.TLabelframe",
            padding=10,
        )
        explainer.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Label(
            explainer,
            text=self._status_reference_text(),
            style="Muted.TLabel",
            wraplength=1220,
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            explainer,
            textvariable=self.secondary_asr_summary,
            style="App.TLabel",
            wraplength=1220,
            justify=tk.LEFT,
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        cards = ttk.Frame(content, style="App.TFrame")
        cards.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        for col in range(6):
            cards.columnconfigure(col, weight=1)

        card_specs = [
            (
                "Всего звонков",
                "total_calls",
                "Сколько аудиозаписей загружено в выбранную БД.",
            ),
            (
                "Распознано",
                "tr_done",
                "Сколько звонков уже прошли ASR и имеют текст.",
            ),
            (
                "Resolve готов",
                "rs_done",
                "Сколько звонков уже прошли склейку и контроль качества текста.",
            ),
            (
                "Анализ готов",
                "an_done",
                "Сколько звонков уже получили итоговый конспект и структурированные поля.",
            ),
            (
                "Ждут 2-й ASR",
                "secondary_asr_pending",
                "Сколько звонков уже распознаны основным ASR, но еще не дотранскрибированы резервным провайдером.",
            ),
            (
                "CRM sync готов",
                "sy_done",
                "Сколько звонков уже синхронизированы в CRM или помечены как завершенные на этапе sync.",
            ),
        ]
        for idx, (caption, key, description) in enumerate(card_specs):
            card = ttk.LabelFrame(cards, text=caption, style="Card.TLabelframe", padding=10)
            card.grid(row=0, column=idx, sticky="nsew", padx=4, pady=4)
            ttk.Label(card, textvariable=self.metric_vars[key], style="CardValue.TLabel").pack(anchor="w")
            ttk.Label(
                card,
                text=description,
                style="CardCaption.TLabel",
                wraplength=210,
                justify=tk.LEFT,
            ).pack(anchor="w", pady=(4, 0))

        stages = ttk.Frame(content, style="App.TFrame")
        stages.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(0, 6))
        stages.columnconfigure(0, weight=1)
        stages.columnconfigure(1, weight=1)
        self._build_status_section(
            stages,
            row=0,
            column=0,
            title="Этап 1. Распознавание",
            rows=[
                ("done", "tr_done", "Звонки уже распознаны. Текст сохранен, этап завершен."),
                ("pending", "tr_pending", "Ждут первого распознавания или повторного запуска этапа."),
                (
                    "2-й ASR pending",
                    "secondary_asr_pending",
                    "Уже распознаны первым ASR, но еще стоят в очереди на дораспознавание резервным провайдером.",
                ),
                ("failed", "tr_failed", "Последняя попытка распознавания упала, но автоповтор еще возможен."),
                ("dead", "tr_dead", "Лимит попыток исчерпан. Нужен ручной requeue или разбор ошибки."),
            ],
        )
        self._build_status_section(
            stages,
            row=0,
            column=1,
            title="Этап 2. Resolve / склейка",
            rows=[
                ("done", "rs_done", "Склейка и улучшение текста завершены, запись готова к анализу."),
                ("pending", "rs_pending", "Ждет запуска resolve после распознавания."),
                ("manual", "rs_manual", "Автоматика не уверена в качестве. Нужна ручная или LLM-проверка."),
                ("skipped", "rs_skipped", "Resolve осознанно пропущен, обычно для коротких или простых звонков."),
                ("failed", "rs_failed", "Resolve упал, но повторная попытка еще возможна."),
                ("dead", "rs_dead", "Resolve исчерпал лимит попыток и остановлен до ручного вмешательства."),
            ],
        )
        self._build_status_section(
            stages,
            row=1,
            column=0,
            title="Этап 3. Анализ разговора",
            rows=[
                ("done", "an_done", "Готов итоговый конспект и структурированные поля для CRM."),
                ("pending", "an_pending", "Ждет запуска анализа после resolve."),
                ("failed", "an_failed", "Последний запуск анализа завершился ошибкой, но может быть повторен."),
                ("dead", "an_dead", "Анализ остановлен после исчерпания попыток."),
            ],
        )
        self._build_status_section(
            stages,
            row=1,
            column=1,
            title="Этап 4. Синхронизация и dead-letter",
            rows=[
                ("sync done", "sy_done", "Синхронизация в CRM завершена."),
                ("sync pending", "sy_pending", "Звонок уже проанализирован, но еще не отправлен в CRM."),
                ("sync failed", "sy_failed", "Последняя попытка sync завершилась ошибкой, но может быть повторена."),
                ("sync dead", "sy_dead", "Синхронизация исчерпала лимит попыток."),
                ("dead transcribe", "dl_transcribe", "Сколько записей сейчас лежит в dead-letter на этапе распознавания."),
                ("dead resolve", "dl_resolve", "Сколько записей застряло в dead-letter на этапе resolve."),
                ("dead analyze", "dl_analyze", "Сколько записей лежит в dead-letter на этапе анализа."),
                ("dead sync", "dl_sync", "Сколько записей лежит в dead-letter на этапе sync."),
            ],
        )

        progress = ttk.LabelFrame(content, text="Прогресс", style="Card.TLabelframe", padding=10)
        progress.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        progress.columnconfigure(1, weight=1)
        ttk.Label(progress, text="Завершено распознавание", style="App.TLabel").grid(
            row=0, column=0, sticky="w", padx=4, pady=4
        )
        self.transcribe_progress = ttk.Progressbar(
            progress, style="Thin.Horizontal.TProgressbar", maximum=100, value=0
        )
        self.transcribe_progress.grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(progress, text="Завершено анализов", style="App.TLabel").grid(
            row=1, column=0, sticky="w", padx=4, pady=4
        )
        self.analyze_progress = ttk.Progressbar(
            progress, style="Thin.Horizontal.TProgressbar", maximum=100, value=0
        )
        self.analyze_progress.grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(progress, text="Обновить сейчас", command=self.on_refresh_stats, style="Primary.TButton").grid(
            row=0, column=2, rowspan=2, padx=6, pady=4, sticky="nsew"
        )

        stats_wrap = ttk.LabelFrame(
            content,
            text="Сырые Stats JSON",
            style="Card.TLabelframe",
            padding=8,
        )
        stats_wrap.grid(row=4, column=0, columnspan=2, sticky="nsew")
        stats_wrap.rowconfigure(0, weight=1)
        stats_wrap.columnconfigure(0, weight=1)
        self.stats_text = tk.Text(stats_wrap, wrap=tk.WORD, height=16)
        self.stats_text.grid(row=0, column=0, sticky="nsew")
        stats_scroll = ttk.Scrollbar(stats_wrap, command=self.stats_text.yview)
        stats_scroll.grid(row=0, column=1, sticky="ns")
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        self._bind_scroll_target(self.stats_text, self.stats_text)
        ttk.Label(
            stats_wrap,
            text=(
                "Ниже показан полный JSON-ответ команды stats. Он полезен для диагностики, "
                "если карточек сверху уже недостаточно."
            ),
            style="Muted.TLabel",
            wraplength=1220,
            justify=tk.LEFT,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=2, pady=(6, 0))

    def _build_status_section(
        self,
        parent: ttk.Frame,
        *,
        row: int,
        column: int,
        title: str,
        rows: list[tuple[str, str, str]],
    ) -> None:
        frame = ttk.LabelFrame(parent, text=title, style="Card.TLabelframe", padding=10)
        frame.grid(row=row, column=column, sticky="nsew", padx=4, pady=4)
        frame.columnconfigure(2, weight=1)
        for idx, (label, key, description) in enumerate(rows):
            ttk.Label(frame, text=label, style="App.TLabel").grid(
                row=idx, column=0, sticky="nw", padx=(0, 8), pady=3
            )
            ttk.Label(frame, textvariable=self.metric_vars[key], style="CardValue.TLabel").grid(
                row=idx,
                column=1,
                sticky="nw",
                padx=(0, 10),
                pady=0,
            )
            ttk.Label(
                frame,
                text=description,
                style="Muted.TLabel",
                wraplength=440,
                justify=tk.LEFT,
            ).grid(row=idx, column=2, sticky="w", pady=3)

    def _status_reference_text(self) -> str:
        return (
            "Эта вкладка показывает состояние текущей базы данных по этапам конвейера. "
            "Каждый этап живет своим статусом: звонок может быть done на распознавании, но pending на resolve или analyze.\n\n"
            "Что означают базовые статусы:\n"
            "pending — этап еще не выполнен или запись была возвращена в очередь.\n"
            "done — этап успешно завершен.\n"
            "failed — последняя попытка завершилась ошибкой, но автоповтор еще возможен.\n"
            "dead — попытки исчерпаны, запись ушла в dead-letter и требует ручного вмешательства.\n"
            "manual — автоматика не уверена в результате и просит ручную проверку.\n"
            "skipped — этап осознанно пропущен по правилам пайплайна.\n\n"
            "Отдельно показывается очередь 2-го ASR: это звонки, где первый провайдер уже отработал, "
            "а резервный еще нет.\n\n"
            "Важно: pending на одном этапе не означает, что звонок не обработан вообще. "
            "Это значит только то, что именно этот этап еще не завершен."
        )

    def _build_functions_tab(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Карта функций", style="Card.TLabelframe", padding=10)
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
            text="Дизайн-ревью (симуляция persona)",
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
        panel = ttk.LabelFrame(parent, text="Лог выполнения", style="Card.TLabelframe", padding=8)
        panel.pack(fill=tk.BOTH, expand=True)
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)
        ttk.Label(
            panel,
            style="Muted.TLabel",
            text="Подсказка: это append-only лог рантайма. Используйте его для контроля длинных прогонов и retry.",
        ).grid(row=0, column=0, sticky="w", padx=2, pady=(0, 6))
        self.log = tk.Text(panel, wrap=tk.WORD)
        self.log.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(panel, command=self.log.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)
        self._bind_scroll_target(self.log, self.log)
        actions = ttk.Frame(panel, style="App.TFrame")
        actions.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(actions, text="Очистить лог", command=self.on_clear_log).pack(side=tk.LEFT)

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
        ttk.Button(frame, text="Выбрать", command=ask_func).grid(
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
        env["RESOLVE_DIALOGUE_MODE"] = self.resolve_dialogue_mode.get().strip()
        env["ANALYZE_PROVIDER"] = self.analyze_provider.get().strip()
        env["DUAL_MERGE_SIMILARITY_THRESHOLD"] = self.merge_similarity_threshold.get().strip()
        env["CODEX_MERGE_MODEL"] = self.codex_merge_model.get().strip()
        env["CODEX_CLI_COMMAND"] = self.codex_cli_command.get().strip()
        env["CODEX_CLI_TIMEOUT_SEC"] = self.codex_cli_timeout_sec.get().strip()
        env["CODEX_REASONING_EFFORT"] = self.codex_reasoning_effort.get().strip()
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

    def _set_batch_progress_idle(self) -> None:
        self.batch_progress.set("Батч: —")

    def _set_batch_progress_message(self, text: str) -> None:
        self.batch_progress.set(text.strip() or "Батч: —")

    def _set_batch_progress(
        self,
        current: int,
        total: int,
        success: int,
        failed: int,
        source_filename: str = "",
    ) -> None:
        base = f"Батч: {current}/{total} (ok {success}, fail {failed})"
        filename = source_filename.strip()
        if filename:
            short_name = filename if len(filename) <= 56 else f"...{filename[-56:]}"
            base = f"{base} · {short_name}"
        self.batch_progress.set(base)

    def _handle_progress_payload(self, payload: dict[str, Any]) -> bool:
        payload_type = str(payload.get("type") or "").strip().lower()
        if payload_type == "transcribe_progress":
            current = int(payload.get("current", 0) or 0)
            total = int(payload.get("total", 0) or 0)
            success = int(payload.get("success", 0) or 0)
            failed = int(payload.get("failed", 0) or 0)
            source_filename = str(payload.get("source_filename") or "")

            self.after(0, self._set_batch_progress, current, total, success, failed, source_filename)

            if total > 0 and current > 0:
                self._append_log(
                    f"[progress] transcribe {current}/{total} (ok={success}, fail={failed}) "
                    f"{source_filename}".strip()
                )
            return True

        if payload_type == "resolve_progress":
            current = int(payload.get("current", 0) or 0)
            total = int(payload.get("total", 0) or 0)
            success = int(payload.get("success", 0) or 0)
            failed = int(payload.get("failed", 0) or 0)
            manual = int(payload.get("manual", 0) or 0)
            skipped_short = int(payload.get("skipped_short", 0) or 0)
            source_filename = str(payload.get("source_filename") or "")
            text = (
                f"Resolve: {current}/{total} (ok {success}, manual {manual}, "
                f"skip {skipped_short}, fail {failed})"
            )
            if source_filename.strip():
                short_name = source_filename if len(source_filename) <= 56 else f"...{source_filename[-56:]}"
                text = f"{text} · {short_name}"
            self.after(0, self._set_batch_progress_message, text)
            if total > 0 and current > 0:
                self._append_log(
                    f"[progress] resolve {current}/{total} (ok={success}, manual={manual}, "
                    f"skip={skipped_short}, fail={failed}) {source_filename}".strip()
                )
            return True

        if payload_type == "pilot_progress":
            current = int(payload.get("current", 0) or 0)
            total = int(payload.get("total", 0) or 0)
            exported = int(payload.get("exported", 0) or 0)
            label = str(payload.get("label") or "").strip()
            stage = str(payload.get("stage") or "pilot").strip()
            source_filename = str(payload.get("source_filename") or payload.get("missing_id") or "")
            text = f"Pilot {stage}: {current}/{total} (exported {exported})"
            if label:
                text = f"{text} [{label}]"
            if source_filename:
                short_name = source_filename if len(source_filename) <= 56 else f"...{source_filename[-56:]}"
                text = f"{text} · {short_name}"
            self.after(0, self._set_batch_progress_message, text)
            if total > 0 and current > 0:
                self._append_log(f"[progress] {text}")
            return True

        return False

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
            line_queue: queue.SimpleQueue[tuple[str, Optional[str]]] = queue.SimpleQueue()
            reader_done = {"stdout": False, "stderr": False}
            stdout_lines: list[str] = []
            stderr_lines: list[str] = []

            def _reader(stream_name: str, stream: Optional[Any]) -> None:
                if stream is None:
                    line_queue.put((stream_name, None))
                    return
                try:
                    for raw_line in stream:
                        line_queue.put((stream_name, raw_line.rstrip("\n")))
                finally:
                    line_queue.put((stream_name, None))

            out_thread = threading.Thread(
                target=_reader,
                args=("stdout", proc.stdout),
                daemon=True,
            )
            err_thread = threading.Thread(
                target=_reader,
                args=("stderr", proc.stderr),
                daemon=True,
            )
            out_thread.start()
            err_thread.start()

            while True:
                drained_any = False
                while True:
                    try:
                        stream_name, line = line_queue.get_nowait()
                    except queue.Empty:
                        break
                    drained_any = True
                    if line is None:
                        reader_done[stream_name] = True
                        continue
                    if stream_name == "stdout":
                        stdout_lines.append(line)
                        if not quiet and line.strip():
                            parsed = self._extract_json_payload(line)
                            if parsed and self._handle_progress_payload(parsed):
                                continue
                            self._append_log(line)
                    else:
                        stderr_lines.append(line)
                        if not quiet and line.strip():
                            self._append_log(line)

                if self.stop_event.is_set() and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()

                if proc.poll() is not None and reader_done["stdout"] and reader_done["stderr"] and not drained_any:
                    break
                time.sleep(0.05)

            out_thread.join(timeout=1)
            err_thread.join(timeout=1)

            while True:
                try:
                    stream_name, line = line_queue.get_nowait()
                except queue.Empty:
                    break
                if line is None:
                    continue
                if stream_name == "stdout":
                    stdout_lines.append(line)
                    if not quiet and line.strip():
                        parsed = self._extract_json_payload(line)
                        if parsed and self._handle_progress_payload(parsed):
                            continue
                        self._append_log(line)
                else:
                    stderr_lines.append(line)
                    if not quiet and line.strip():
                        self._append_log(line)

            stdout = "\n".join(stdout_lines)
            stderr = "\n".join(stderr_lines)
            if stderr and not quiet:
                lowered = stderr.lower()
                if "no module named 'sqlalchemy'" in lowered or 'no module named "sqlalchemy"' in lowered:
                    self._append_log(
                        "[hint] Выбранный Backend Python не содержит зависимости проекта. "
                        "Укажите stable runtime python: stable_runtime/venv_stable/bin/python "
                        "или запускайте UI через ./stable_runtime/run-ui.sh"
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

    def _pipeline_primary_env(self) -> dict[str, str]:
        primary = self.transcribe_provider.get().strip() or "mlx"
        return {
            "TRANSCRIBE_PROVIDER": primary,
            "DUAL_TRANSCRIBE_ENABLED": "0",
            "SECONDARY_TRANSCRIBE_PROVIDER": "",
        }

    def _pipeline_stage_env(self) -> dict[str, str]:
        primary = self.transcribe_provider.get().strip() or "mlx"
        secondary = self.secondary_provider.get().strip()
        dual_enabled = bool(secondary and secondary != primary)
        return {
            "TRANSCRIBE_PROVIDER": primary,
            "DUAL_TRANSCRIBE_ENABLED": "1" if dual_enabled else "0",
            "SECONDARY_TRANSCRIBE_PROVIDER": secondary if dual_enabled else "",
        }

    def _pipeline_secondary_env(self) -> dict[str, str]:
        return self._pipeline_stage_env()

    def _selected_pipeline_stages(self) -> list[str]:
        stages: list[str] = []
        if self.pipeline_stage_transcribe.get():
            stages.append("transcribe")
        if self.pipeline_stage_backfill.get():
            stages.append("backfill-second-asr")
        if self.pipeline_stage_resolve.get():
            stages.append("resolve")
        if self.pipeline_stage_analyze.get():
            stages.append("analyze")
        if self.pipeline_stage_sync.get():
            stages.append("sync")
        return stages

    def _launch_task(self, title: str, fn) -> None:
        if self.active_task and self.active_task.is_alive():
            self._append_log(f"[busy] уже выполняется другая задача: {self.run_state.get()}")
            return

        self.stop_event.clear()
        self._append_log(f"\n=== {title} ===")
        self._set_run_state(title)
        self._set_batch_progress_idle()

        def _runner() -> None:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                self._append_log(f"[error] {exc}")
            finally:
                self._append_log(f"=== {title}: finished ===")
                self.after(0, self._set_run_state, "Ожидание")
                self.after(0, self._set_batch_progress_idle)
                self.after(0, self._refresh_stats_async, True)

        self.active_task = threading.Thread(target=_runner, daemon=True)
        self.active_task.start()

    def _format_payload_inline(self, payload: dict[str, Any]) -> str:
        ordered = [
            "processed",
            "success",
            "failed",
            "partial",
            "exhausted",
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
        rc, payload = self._run_cli(
            ["stats"],
            env_overrides=self._pipeline_stage_env(),
            quiet=silent,
        )
        if rc != 0 or not payload:
            if not silent:
                self._append_log("[stats] не удалось получить статус.")
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
        sy = payload.get("sync_status") or {}
        dl = payload.get("dead_letter_stage") or {}
        secondary_backfill = payload.get("secondary_asr_backfill") or {}

        self.metric_vars["total_calls"].set(str(total))
        self.metric_vars["secondary_asr_pending"].set(
            str(int(secondary_backfill.get("pending", 0) or 0))
        )
        self.metric_vars["tr_done"].set(str(int(tr.get("done", 0) or 0)))
        self.metric_vars["tr_pending"].set(str(int(tr.get("pending", 0) or 0)))
        self.metric_vars["tr_failed"].set(str(int(tr.get("failed", 0) or 0)))
        self.metric_vars["tr_dead"].set(str(int(tr.get("dead", 0) or 0)))

        self.metric_vars["rs_done"].set(str(int(rs.get("done", 0) or 0)))
        self.metric_vars["rs_skipped"].set(str(int(rs.get("skipped", 0) or 0)))
        self.metric_vars["rs_manual"].set(str(int(rs.get("manual", 0) or 0)))
        self.metric_vars["rs_pending"].set(str(int(rs.get("pending", 0) or 0)))
        self.metric_vars["rs_failed"].set(str(int(rs.get("failed", 0) or 0)))
        self.metric_vars["rs_dead"].set(str(int(rs.get("dead", 0) or 0)))

        self.metric_vars["an_done"].set(str(int(an.get("done", 0) or 0)))
        self.metric_vars["an_pending"].set(str(int(an.get("pending", 0) or 0)))
        self.metric_vars["an_failed"].set(str(int(an.get("failed", 0) or 0)))
        self.metric_vars["an_dead"].set(str(int(an.get("dead", 0) or 0)))
        self.metric_vars["sy_done"].set(str(int(sy.get("done", 0) or 0)))
        self.metric_vars["sy_pending"].set(str(int(sy.get("pending", 0) or 0)))
        self.metric_vars["sy_failed"].set(str(int(sy.get("failed", 0) or 0)))
        self.metric_vars["sy_dead"].set(str(int(sy.get("dead", 0) or 0)))
        self.metric_vars["dl_transcribe"].set(str(int(dl.get("transcribe", 0) or 0)))
        self.metric_vars["dl_resolve"].set(str(int(dl.get("resolve", 0) or 0)))
        self.metric_vars["dl_analyze"].set(str(int(dl.get("analyze", 0) or 0)))
        self.metric_vars["dl_sync"].set(str(int(dl.get("sync", 0) or 0)))
        backfill_enabled = bool(secondary_backfill.get("enabled"))
        backfill_primary = str(secondary_backfill.get("primary_provider") or "").strip() or "—"
        backfill_secondary = str(secondary_backfill.get("secondary_provider") or "").strip() or "—"
        backfill_pending = int(secondary_backfill.get("pending", 0) or 0)
        backfill_retry_pending = int(secondary_backfill.get("retry_pending", 0) or 0)
        backfill_exhausted = int(secondary_backfill.get("exhausted", 0) or 0)
        if backfill_enabled:
            self.secondary_asr_summary.set(
                "Очередь 2-го ASR: "
                f"{backfill_primary} -> {backfill_secondary}, "
                f"ждут дораспознавания: {backfill_pending}, "
                f"повторные retry: {backfill_retry_pending}, "
                f"исчерпаны: {backfill_exhausted}"
            )
        else:
            self.secondary_asr_summary.set(
                "Очередь 2-го ASR: отключена для текущей конфигурации распознавания"
            )

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
        self._launch_task("Инициализация БД", lambda: self._run_cli(["init-db"]))

    def on_ingest(self) -> None:
        def _task() -> None:
            args = ["ingest", "--recordings-dir", self.recordings_dir.get().strip()]
            meta = self.metadata_csv.get().strip()
            if meta:
                args.extend(["--metadata-csv", meta])
            self._run_cli(args)
            self._refresh_stats_sync(silent=True)

        self._launch_task("Загрузка звонков", _task)

    def on_refresh_stats(self) -> None:
        self._launch_task("Обновление статуса", lambda: self._refresh_stats_sync(silent=False))

    def on_transcribe_batch(self) -> None:
        def _task() -> None:
            env = self._transcribe_mode_env()
            self._run_cli(["transcribe", "--limit", str(self._limit_value())], env_overrides=env)
            self._refresh_stats_sync(silent=True)

        self._launch_task("Этап 1: Батч распознавания", _task)

    def on_transcribe_drain(self) -> None:
        def _task() -> None:
            env = self._transcribe_mode_env()
            summary = self._drain_stage("transcribe", env_overrides=env)
            self._append_log(f"[transcribe drain] сводка: {summary}")
            self._refresh_stats_sync(silent=True)

        self._launch_task("Этап 1: Распознать все pending", _task)

    def on_fill_missing_second_asr(self) -> None:
        def _task() -> None:
            primary = self.transcribe_provider.get().strip() or "mlx"
            secondary = self.secondary_provider.get().strip() or "gigaam"
            env = {
                "TRANSCRIBE_PROVIDER": primary,
                "DUAL_TRANSCRIBE_ENABLED": "1",
                "SECONDARY_TRANSCRIBE_PROVIDER": secondary,
            }
            summary = self._drain_stage("backfill-second-asr", env_overrides=env)
            self._append_log(f"[fill-missing-second-asr] transcribe summary: {summary}")
            self._refresh_stats_sync(silent=True)

        self._launch_task("Этап 1: Дораспознать недостающий 2-й ASR", _task)

    def on_pipeline_drain(self) -> None:
        def _task() -> None:
            stages = self._selected_pipeline_stages()
            if not stages:
                self._append_log("[pipeline] не выбран ни один этап")
                return

            self._append_log(f"[pipeline] stages={','.join(stages)}")
            for stage in stages:
                if self.stop_event.is_set():
                    break
                if stage == "transcribe":
                    summary = self._drain_stage(
                        "transcribe",
                        env_overrides=self._pipeline_primary_env(),
                    )
                elif stage == "backfill-second-asr":
                    summary = self._drain_stage(
                        "backfill-second-asr",
                        env_overrides=self._pipeline_secondary_env(),
                    )
                elif stage == "resolve":
                    summary = self._drain_stage(
                        "resolve",
                        env_overrides={
                            **self._pipeline_stage_env(),
                            "RESOLVE_LLM_PROVIDER": self.resolve_llm_provider.get().strip()
                        },
                    )
                elif stage == "analyze":
                    summary = self._drain_stage(
                        "analyze",
                        env_overrides={
                            **self._pipeline_stage_env(),
                            "ANALYZE_PROVIDER": self.analyze_provider.get().strip()
                        },
                    )
                elif stage == "sync":
                    summary = self._drain_stage(
                        "sync",
                        env_overrides=self._pipeline_stage_env(),
                    )
                else:
                    self._append_log(f"[pipeline] неизвестный этап: {stage}")
                    continue
                self._append_log(f"[pipeline {stage}] сводка: {summary}")
            self._refresh_stats_sync(silent=True)

        self._launch_task("Конвейер: обработать выбранные этапы", _task)

    def on_codex_resolve_batch(self) -> None:
        def _task() -> None:
            env = {"RESOLVE_LLM_PROVIDER": "codex_cli"}
            self._run_cli(["resolve", "--limit", str(self._limit_value())], env_overrides=env)
            self._refresh_stats_sync(silent=True)

        self._launch_task("Codex Resolve (батч)", _task)

    def on_codex_analyze_batch(self) -> None:
        def _task() -> None:
            env = {"ANALYZE_PROVIDER": "codex_cli"}
            self._run_cli(["analyze", "--limit", str(self._limit_value())], env_overrides=env)
            self._refresh_stats_sync(silent=True)

        self._launch_task("Codex Analyze (батч)", _task)

    def on_codex_postprocess_drain(self) -> None:
        def _task() -> None:
            resolve_summary = self._drain_stage(
                "resolve",
                env_overrides={"RESOLVE_LLM_PROVIDER": "codex_cli"},
            )
            self._append_log(f"[codex resolve drain] сводка: {resolve_summary}")
            if self.stop_event.is_set():
                return
            analyze_summary = self._drain_stage(
                "analyze",
                env_overrides={"ANALYZE_PROVIDER": "codex_cli"},
            )
            self._append_log(f"[codex analyze drain] сводка: {analyze_summary}")
            self._refresh_stats_sync(silent=True)

        self._launch_task("Codex: обработать все pending", _task)

    def on_prepare_resolve_pilot(self) -> None:
        def _task() -> None:
            args = [
                "prepare-resolve-pilot",
                "--limit",
                str(self._limit_value()),
                "--seed",
                self.pilot_seed.get().strip() or "42",
                "--ids-out",
                self.pilot_ids_path.get().strip(),
            ]
            self._run_cli(args)
            self._refresh_stats_sync(silent=True)

        self._launch_task("Пилот Resolve: подготовить выборку", _task)

    def _export_resolve_pilot_bundle(self, label: str) -> None:
        args = [
            "export-pilot-bundle",
            "--ids-in",
            self.pilot_ids_path.get().strip(),
            "--out",
            self.pilot_export_dir.get().strip(),
            "--label",
            label,
        ]
        self._run_cli(args)

    def on_export_resolve_pilot_initial(self) -> None:
        self._launch_task(
            "Пилот Resolve: экспорт initial",
            lambda: self._export_resolve_pilot_bundle("initial"),
        )

    def on_export_resolve_pilot_final(self) -> None:
        self._launch_task(
            "Пилот Resolve: экспорт final",
            lambda: self._export_resolve_pilot_bundle("final"),
        )

    def on_prepare_and_export_resolve_pilot_initial(self) -> None:
        def _task() -> None:
            self._run_cli(
                [
                    "prepare-resolve-pilot",
                    "--limit",
                    str(self._limit_value()),
                    "--seed",
                    self.pilot_seed.get().strip() or "42",
                    "--ids-out",
                    self.pilot_ids_path.get().strip(),
                ]
            )
            if self.stop_event.is_set():
                return
            self._export_resolve_pilot_bundle("initial")
            self._refresh_stats_sync(silent=True)

        self._launch_task("Пилот Resolve: подготовить + initial", _task)

    def on_resolve_batch(self) -> None:
        self._launch_task(
            "Resolve батч (текущий провайдер)",
            lambda: self._run_cli(["resolve", "--limit", str(self._limit_value())]),
        )

    def on_analyze_batch(self) -> None:
        self._launch_task(
            "Analyze батч (текущий провайдер)",
            lambda: self._run_cli(["analyze", "--limit", str(self._limit_value())]),
        )

    def on_sync_batch(self) -> None:
        self._launch_task(
            "Sync батч",
            lambda: self._run_cli(["sync", "--limit", str(self._limit_value())]),
        )

    def on_worker_once(self) -> None:
        stages = self._selected_pipeline_stages()
        if not stages:
            self._append_log("[worker] не выбран ни один этап")
            return
        self._launch_task(
            "Worker один цикл",
            lambda: self._run_cli(
                [
                    "worker",
                    "--once",
                    "--stage-limit",
                    str(self._limit_value()),
                    "--stages",
                    ",".join(stages),
                ],
                env_overrides=self._pipeline_stage_env(),
            ),
        )

    def on_worker_start(self) -> None:
        if self.worker_process and self.worker_process.poll() is None:
            self._append_log("[worker] уже запущен")
            return
        stages = self._selected_pipeline_stages()
        if not stages:
            self._append_log("[worker] не выбран ни один этап")
            return

        cmd = [
            self._backend_python_exe(),
            "-u",
            "-m",
            "mango_mvp.cli",
            "worker",
            "--stage-limit",
            str(self._limit_value()),
            "--stages",
            ",".join(stages),
            "--poll-sec",
            self.worker_poll_sec.get().strip() or "10",
            "--max-idle-cycles",
            self.worker_max_idle_cycles.get().strip() or "30",
        ]
        self._append_log("\n=== Worker старт ===")
        self._append_log(f"$ {' '.join(cmd)}")
        try:
            self.worker_process = subprocess.Popen(
                cmd,
                cwd=self.project_dir.get().strip() or str(Path.cwd()),
                env=self._build_env(self._pipeline_stage_env()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"[worker] ошибка запуска: {exc}")
            self.worker_process = None
            return

        def _reader() -> None:
            assert self.worker_process is not None
            if self.worker_process.stdout:
                for line in self.worker_process.stdout:
                    self._append_log(line.rstrip())
            code = self.worker_process.wait()
            self._append_log(f"[worker] завершился с кодом={code}")

        threading.Thread(target=_reader, daemon=True).start()

    def on_worker_stop(self) -> None:
        if not self.worker_process or self.worker_process.poll() is not None:
            self._append_log("[worker] не запущен")
            return
        self.worker_process.terminate()
        self._append_log("[worker] отправлен запрос на остановку")

    def on_requeue_dead(self) -> None:
        self._launch_task("Вернуть dead в очередь", lambda: self._run_cli(["requeue-dead", "--stage", "all"]))

    def on_reset_missing_variants(self) -> None:
        self._launch_task(
            "Сбросить missing variants",
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
            "Экспорт очереди ручной проверки",
            lambda: self._run_cli(
                ["export-review-queue", "--out", str(out_path), "--limit", "100000"]
            ),
        )

    def on_export_failed_resolve_queue(self) -> None:
        out_path = Path(self.project_dir.get()) / "failed_resolve_queue.csv"
        self._launch_task(
            "Экспорт failed resolve",
            lambda: self._run_cli(
                ["export-failed-resolve-queue", "--out", str(out_path), "--limit", "100000"]
            ),
        )

    def on_export_crm_fields(self) -> None:
        out_path = Path(self.project_dir.get()) / "crm_fields.csv"
        self._launch_task(
            "Экспорт полей CRM",
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
            self._append_log("[task] отправлен запрос на остановку текущего CLI-процесса.")
        else:
            self._append_log("[task] активный CLI-процесс не найден.")

    def on_clear_log(self) -> None:
        self.log.delete("1.0", tk.END)

    def _functions_reference_text(self) -> str:
        return (
            "Mango Calls Studio - справка по функциям\n\n"
            "Рекомендуемый порядок\n"
            "1. Инициализировать БД\n"
            "   Создает схему в выбранном SQLite-файле.\n\n"
            "2. Загрузить звонки\n"
            "   Сканирует папку записей, парсит метаданные из имен Mango и добавляет только новые файлы.\n"
            "   Уже индексированные source_file автоматически пропускаются.\n\n"
            "3. Этап 1: Только распознавание\n"
            "   Выберите режим и размер батча, затем нажмите:\n"
            "   - Этап 1: Батч распознавания\n"
            "   - Этап 1: Распознать все pending\n"
            "   - Этап 1: Дораспознать 2-й ASR\n"
            "   Режимы:\n"
            "   - whisper: только MLX Whisper\n"
            "   - gigaam: только GigaAM\n"
            "   - dual: Whisper + GigaAM со склейкой\n"
            "   Resume-safe: обрабатываются только pending+failed; done сохраняются.\n\n"
            "4. Этап 2: Постобработка Codex\n"
            "   Пайплайн постобработки через Codex CLI:\n"
            "   - Resolve улучшает рискованные транскрипты и флаги качества\n"
            "   - Analyze собирает CRM-готовые конспекты и структурированные поля\n"
            "   - Поле Codex reasoning управляет глубиной рассуждений Codex CLI для массовой обработки\n"
            "   Команды постобработки независимы от этапа распознавания.\n\n"
            "5. Конвейерные этапы\n"
            "   Чекбоксы Whisper / GigaAM / Resolve / Analyze / Sync управляют:\n"
            "   - кнопкой 'Конвейер: обработать выбранные этапы'\n"
            "   - Worker один цикл\n"
            "   - Worker старт\n"
            "   Это позволяет, например, остановить конвейер на Resolve и пока не запускать Analyze.\n\n"
            "6. Codex: обработать все pending\n"
            "   Сначала полностью проходит очередь resolve, затем analyze до processed=0.\n\n"
            "Операционные и восстановительные инструменты\n"
            "- Worker старт/стоп: непрерывный устойчивый цикл обработки только для выбранных этапов.\n"
            "- Вернуть dead в очередь: переводит dead-letter записи обратно в pending.\n"
            "- Сбросить missing variants: повторно открывает done-транскрипты без dual-ASR вариантов.\n"
            "- Экспорт очереди проверки: CSV для ручного QA.\n"
            "- Экспорт failed resolve: CSV по failed/dead вызовам resolve.\n"
            "- Экспорт полей CRM: выгружает поля анализа в CSV-колонки.\n\n"
            "Гарантия идемпотентности\n"
            "- Этапы используют статусные колонки в БД.\n"
            "- Строки со статусом done не обрабатываются обычными командами повторно.\n"
            "- Повторная работа возможна только при явном сбросе статусов.\n\n"
            "Где хранятся результаты\n"
            "- Статусы и payload хранятся в выбранной SQLite-базе.\n"
            "- Транскрипты и анализ экспортируются в выбранную папку.\n"
        )

    def _design_review_text(self) -> str:
        return (
            "Заметки симулированного дизайн-ревью (не реальный сотрудник Apple)\n\n"
            "Итерация 1 - ясность\n"
            "- Проблема: старый UI смешивал слишком много контролов в одной длинной панели.\n"
            "- Изменение: добавлены вкладки (Конвейер / Статус / Справка / Дизайн-ревью).\n"
            "- Результат: пользователь выполняет этапы без лишнего технического шума.\n\n"
            "Итерация 2 - операционная уверенность\n"
            "- Проблема: было неясно, могут ли готовые звонки обработаться повторно.\n"
            "- Изменение: явные подсказки про resume-safe, карточки статусов, прогресс-бары и queue drain.\n"
            "- Результат: оператор видит pending/done/failed и продолжает с точного чекпоинта.\n\n"
            "Итерация 3 - разделение этапов\n"
            "- Проблема: этап 1 (ASR) и этап 2 (Codex) визуально смешивались.\n"
            "- Изменение: отдельные блоки действий для Этапа 1 и Этапа 2.\n"
            "- Результат: поток \"только распознавание\" запускается интуитивно.\n\n"
            "Итерация 4 - доступность и прокрутка\n"
            "- Проблема: на небольшом окне часть контролов обрезалась.\n"
            "- Изменение: добавлены прокручиваемые контейнеры вкладок с поддержкой колеса мыши.\n"
            "- Результат: все элементы доступны без ручного ресайза окна.\n\n"
            "Итерация 5 - устранение двусмысленности dual режима\n"
            "- Проблема: dual toggle с крестиком/галочкой выглядел неоднозначно.\n"
            "- Изменение: убран неоднозначный toggle; режим определяется только полем transcribe mode.\n"
            "- Результат: один явный источник истины, без конфликтующих состояний.\n\n"
            "Вердикт ревьюера\n"
            "- Иерархия информации стала понятной.\n"
            "- Основные действия вынесены на первый экран.\n"
            "- Контроль статусов удобен для длинных прогонов.\n"
            "- Текущий UI подходит для полу-боевой эксплуатации.\n"
        )


def main() -> int:
    app = MangoMvpGui()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
