from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .config import AppConfig
from .file_io import (
    FileValidationError,
    default_output_path,
    validate_input_file,
    validate_output_path,
)
from .transcriber import (
    LocalTranscriber,
    TranscriptionCancelled,
    TranscriptionResult,
)


class SpeechToTextApp:
    def __init__(
        self,
        root: tk.Tk,
        config: AppConfig,
        transcriber: LocalTranscriber,
    ) -> None:
        self.root = root
        self.config = config
        self.transcriber = transcriber
        self._logger = logging.getLogger(__name__)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.status_var = tk.StringVar(value="请选择一个音频文件。")
        self.detail_var = tk.StringVar(value="识别过程在本机完成，音频不会上传。")

        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._cancel_event: threading.Event | None = None
        self._running = False
        self._closing = False
        self._progress_is_indeterminate = False
        self._setting_output_programmatically = False
        self._output_is_automatic = True

        self._build_window()
        self.output_var.trace_add("write", self._on_output_edited)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_events)

    def _build_window(self) -> None:
        self.root.title(self.config.app_name)
        self.root.geometry("760x330")
        self.root.minsize(680, 310)

        style = ttk.Style(self.root)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 17, "bold"))
        style.configure("Hint.TLabel", foreground="#5f6368")
        style.configure("Status.TLabel", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Primary.TButton", font=("Microsoft YaHei UI", 10, "bold"))

        container = ttk.Frame(self.root, padding=(24, 20, 24, 20))
        container.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)

        ttk.Label(container, text="本地音频转文字", style="Title.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 18)
        )

        ttk.Label(container, text="输入音频").grid(
            row=1, column=0, sticky="w", padx=(0, 12), pady=7
        )
        self.input_entry = ttk.Entry(container, textvariable=self.input_var)
        self.input_entry.grid(row=1, column=1, sticky="ew", pady=7)
        self.input_button = ttk.Button(
            container, text="选择音频", command=self._choose_input
        )
        self.input_button.grid(row=1, column=2, padx=(10, 0), pady=7)

        ttk.Label(container, text="输出 TXT").grid(
            row=2, column=0, sticky="w", padx=(0, 12), pady=7
        )
        self.output_entry = ttk.Entry(container, textvariable=self.output_var)
        self.output_entry.grid(row=2, column=1, sticky="ew", pady=7)
        self.output_button = ttk.Button(
            container, text="更改路径", command=self._choose_output
        )
        self.output_button.grid(row=2, column=2, padx=(10, 0), pady=7)

        action_row = ttk.Frame(container)
        action_row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(15, 8))
        action_row.columnconfigure(0, weight=1)
        self.start_button = ttk.Button(
            action_row,
            text="开始转写",
            command=self._on_primary_action,
            style="Primary.TButton",
        )
        self.start_button.grid(row=0, column=0, sticky="ew", ipady=4)

        self.progress = ttk.Progressbar(
            container, mode="determinate", maximum=100, value=0
        )
        self.progress.grid(
            row=4, column=0, columnspan=3, sticky="ew", pady=(10, 9)
        )

        ttk.Label(container, textvariable=self.status_var, style="Status.TLabel").grid(
            row=5, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(
            container,
            textvariable=self.detail_var,
            style="Hint.TLabel",
            wraplength=690,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(5, 0))

    def _choose_input(self) -> None:
        patterns = " ".join(f"*{suffix}" for suffix in self.config.supported_extensions)
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="选择音频文件",
            filetypes=(("音频文件", patterns), ("所有文件", "*.*")),
        )
        if not selected:
            return

        self.input_var.set(selected)
        if self._output_is_automatic or not self.output_var.get().strip():
            self._set_output_value(str(default_output_path(selected)), automatic=True)
        self.status_var.set("已选择音频，可以开始转写。")

    def _choose_output(self) -> None:
        input_path = self.input_var.get().strip()
        suggested = default_output_path(input_path) if input_path else Path("转写结果.txt")
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="选择文本保存位置",
            initialdir=str(suggested.parent),
            initialfile=suggested.name,
            defaultextension=".txt",
            filetypes=(("文本文件", "*.txt"),),
        )
        if selected:
            self._set_output_value(selected, automatic=False)

    def _on_output_edited(self, *_args: object) -> None:
        if not self._setting_output_programmatically:
            self._output_is_automatic = False

    def _set_output_value(self, value: str, *, automatic: bool) -> None:
        self._setting_output_programmatically = True
        try:
            self.output_var.set(value)
        finally:
            self._setting_output_programmatically = False
        self._output_is_automatic = automatic

    def _on_primary_action(self) -> None:
        if self._running:
            self._request_cancel()
        else:
            self._start_transcription()

    def _start_transcription(self) -> None:
        try:
            input_path = validate_input_file(
                self.input_var.get().strip(), self.config.supported_extensions
            )
            output_text = self.output_var.get().strip()
            if not output_text:
                output_text = str(default_output_path(input_path))
                self._set_output_value(output_text, automatic=True)
            output_path = validate_output_path(output_text)
            if input_path.resolve() == output_path.resolve():
                raise FileValidationError("输出文件不能覆盖输入音频。")
        except FileValidationError as exc:
            messagebox.showwarning("无法开始", str(exc), parent=self.root)
            return

        if str(output_path) != self.output_var.get():
            self._set_output_value(
                str(output_path), automatic=self._output_is_automatic
            )

        if output_path.exists() and not messagebox.askyesno(
            "确认覆盖",
            f"输出文件已存在：\n{output_path}\n\n是否在转写成功后覆盖它？",
            parent=self.root,
        ):
            return

        allow_download = False
        if not self.transcriber.model_is_available():
            allow_download = messagebox.askokcancel(
                "首次使用需要下载模型",
                "本机尚未安装语音识别模型。首次使用需要联网下载约数百 MB "
                "的数据；下载完成后即可断网使用。\n\n"
                f"模型将保存在：\n{self.config.model_dir}\n\n"
                "音频和转写文本不会上传。是否下载并继续？",
                parent=self.root,
            )
            if not allow_download:
                return

        self._running = True
        self._cancel_event = threading.Event()
        self._set_running_controls(True)
        self.status_var.set("正在准备转写……")
        self.detail_var.set("首次加载模型可能需要一些时间，请勿关闭程序。")
        self._show_indeterminate_progress()

        self._worker_thread = threading.Thread(
            target=self._worker,
            args=(input_path, output_path, allow_download, self._cancel_event),
            name="transcription-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def _worker(
        self,
        input_path: Path,
        output_path: Path,
        allow_download: bool,
        cancel_event: threading.Event,
    ) -> None:
        try:
            result = self.transcriber.transcribe(
                input_path,
                output_path,
                cancel_event,
                allow_download=allow_download,
                status_callback=lambda message: self._events.put(("status", message)),
                progress_callback=lambda value: self._events.put(("progress", value)),
            )
        except TranscriptionCancelled as exc:
            self._events.put(("cancelled", str(exc)))
        except Exception as exc:
            self._logger.error("Background transcription failed", exc_info=True)
            self._events.put(
                ("error", str(exc) or "转写失败，请查看本地日志。")
            )
        else:
            self._events.put(("completed", result))

    def _request_cancel(self) -> None:
        if self._cancel_event is None or self._cancel_event.is_set():
            return
        self._cancel_event.set()
        self.start_button.configure(state="disabled")
        self.status_var.set("正在取消……")
        self.detail_var.set("模型加载或当前识别片段结束后，任务将安全停止。")

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self._events.get_nowait()
                self._handle_event(event, payload)
        except queue.Empty:
            pass

        if self._closing and not self._running:
            self.root.destroy()
            return
        self.root.after(100, self._poll_events)

    def _handle_event(self, event: str, payload: object) -> None:
        if event == "status":
            message = str(payload)
            self.status_var.set(message)
            if message.startswith(("正在下载", "正在加载", "GPU", "正在分析")):
                self._show_indeterminate_progress()
            return

        if event == "progress":
            self._show_determinate_progress(float(payload))
            return

        if event == "completed":
            self._finish_completed(payload)
            return

        if event == "cancelled":
            self._finish_task(progress_value=0.0)
            self.status_var.set("转写已取消。")
            self.detail_var.set("未生成或覆盖输出文件。")
            return

        if event == "error":
            self._finish_task(progress_value=0.0)
            self.status_var.set("转写失败。")
            self.detail_var.set("详细信息已写入本地日志。")
            if not self._closing:
                messagebox.showerror("转写失败", str(payload), parent=self.root)

    def _finish_completed(self, payload: object) -> None:
        result = payload
        if not isinstance(result, TranscriptionResult):
            self._finish_task(progress_value=0.0)
            return

        self._finish_task(progress_value=100.0)
        elapsed = self._format_duration(result.elapsed_seconds)
        self.status_var.set("转写完成。")
        self.detail_var.set(
            f"语言：{result.language}　设备：{result.device}　耗时：{elapsed}"
        )
        if self._closing:
            return

        messagebox.showinfo(
            "转写完成",
            f"文本已保存到：\n{result.output_path}\n\n"
            f"识别语言：{result.language}\n"
            f"耗时：{elapsed}",
            parent=self.root,
        )

    def _finish_task(self, *, progress_value: float) -> None:
        self._running = False
        self._worker_thread = None
        self._cancel_event = None
        self._set_running_controls(False)
        self._show_determinate_progress(progress_value)

    def _set_running_controls(self, running: bool) -> None:
        entry_state = "disabled" if running else "normal"
        button_state = "disabled" if running else "normal"
        self.input_entry.configure(state=entry_state)
        self.output_entry.configure(state=entry_state)
        self.input_button.configure(state=button_state)
        self.output_button.configure(state=button_state)
        self.start_button.configure(
            text="取消转写" if running else "开始转写",
            state="normal",
        )

    def _show_indeterminate_progress(self) -> None:
        if self._progress_is_indeterminate:
            return
        self.progress.stop()
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self._progress_is_indeterminate = True

    def _show_determinate_progress(self, value: float) -> None:
        if self._progress_is_indeterminate:
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self._progress_is_indeterminate = False
        self.progress.configure(value=max(0.0, min(100.0, value)))

    def _on_close(self) -> None:
        if not self._running:
            self.root.destroy()
            return
        if self._closing:
            return
        if not messagebox.askyesno(
            "停止转写并退出",
            "转写仍在进行。是否安全停止任务并退出？",
            parent=self.root,
        ):
            return

        self._closing = True
        self._request_cancel()
        self.input_button.configure(state="disabled")
        self.output_button.configure(state="disabled")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        minutes, remaining_seconds = divmod(total_seconds, 60)
        if minutes:
            return f"{minutes} 分 {remaining_seconds} 秒"
        return f"{remaining_seconds} 秒"
