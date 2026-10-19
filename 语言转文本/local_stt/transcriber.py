from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import AppConfig
from .file_io import (
    AtomicTextWriter,
    FileValidationError,
    OutputWriteError,
    validate_input_file,
    validate_output_path,
)


StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[float], None]
ModelFactory = Callable[..., Any]


class TranscriptionError(RuntimeError):
    """Base error shown to the user when transcription cannot complete."""


class ModelDownloadRequired(TranscriptionError):
    """The local model is not present and downloading was not approved."""


class ModelLoadError(TranscriptionError):
    """The speech recognition model could not be loaded."""


class AudioDecodeError(TranscriptionError):
    """The selected audio file could not be decoded."""


class NoSpeechDetectedError(TranscriptionError):
    """The audio did not contain any recognized speech."""


class TranscriptionCancelled(TranscriptionError):
    """The current transcription was cancelled by the user."""


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    output_path: Path
    language: str
    language_probability: float
    audio_duration: float
    elapsed_seconds: float
    character_count: int
    device: str


_LANGUAGE_NAMES = {
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
    "ko": "韩文",
    "fr": "法文",
    "de": "德文",
    "es": "西班牙文",
    "ru": "俄文",
    "pt": "葡萄牙文",
    "it": "意大利文",
}


class LocalTranscriber:
    def __init__(
        self,
        config: AppConfig,
        model_factory: ModelFactory | None = None,
        device_candidates: Iterable[tuple[str, str]] | None = None,
    ) -> None:
        self.config = config
        self._model_factory = model_factory
        self._configured_candidates = (
            tuple(device_candidates) if device_candidates is not None else None
        )
        self._model: Any | None = None
        self._device_label = ""
        self._force_cpu = False
        self._model_lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

    def model_is_available(self) -> bool:
        required_files = ("model.bin", "config.json", "tokenizer.json")
        return all(
            (self.config.model_dir / filename).is_file()
            and (self.config.model_dir / filename).stat().st_size > 0
            for filename in required_files
        )

    def transcribe(
        self,
        input_path: str | Path,
        output_path: str | Path,
        cancel_event: threading.Event,
        *,
        allow_download: bool,
        status_callback: StatusCallback | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        status = status_callback or (lambda _message: None)
        progress = progress_callback or (lambda _value: None)
        started_at = time.monotonic()

        source = validate_input_file(
            input_path, supported_extensions=self.config.supported_extensions
        )
        destination = validate_output_path(output_path)
        if source.resolve() == destination.resolve():
            raise FileValidationError("输出文件不能覆盖输入音频。")
        self._raise_if_cancelled(cancel_event)

        model, device_label = self._load_model(allow_download, status)
        self._raise_if_cancelled(cancel_event)

        status("正在分析音频并自动检测语言……")
        progress(0.0)

        try:
            segments, info = model.transcribe(
                str(source),
                language=None,
                task="transcribe",
                beam_size=self.config.beam_size,
                vad_filter=self.config.vad_filter,
                word_timestamps=False,
                condition_on_previous_text=True,
            )

            language = str(getattr(info, "language", "unknown") or "unknown")
            language_probability = float(
                getattr(info, "language_probability", 0.0) or 0.0
            )
            duration = float(getattr(info, "duration", 0.0) or 0.0)
            language_name = _LANGUAGE_NAMES.get(language, language)
            status(
                f"检测语言：{language_name} ({language})，"
                f"置信度 {language_probability:.0%}"
            )

            last_percent = -1
            with AtomicTextWriter(destination) as writer:
                for segment in segments:
                    self._raise_if_cancelled(cancel_event)
                    writer.write_segment(str(getattr(segment, "text", "")))

                    segment_end = float(getattr(segment, "end", 0.0) or 0.0)
                    if duration > 0:
                        percent = min(99, max(0, int(segment_end / duration * 100)))
                        if percent > last_percent:
                            progress(float(percent))
                            last_percent = percent

                self._raise_if_cancelled(cancel_event)
                if writer.line_count == 0:
                    raise NoSpeechDetectedError(
                        "未检测到可识别语音，未生成输出文件。"
                    )
                writer.commit()
                character_count = writer.character_count

            progress(100.0)
            elapsed = time.monotonic() - started_at
            self._logger.info(
                "Transcription completed: language=%s duration=%.2fs elapsed=%.2fs "
                "characters=%d device=%s output=%s",
                language,
                duration,
                elapsed,
                character_count,
                device_label,
                destination,
            )
            return TranscriptionResult(
                output_path=destination,
                language=language,
                language_probability=language_probability,
                audio_duration=duration,
                elapsed_seconds=elapsed,
                character_count=character_count,
                device=device_label,
            )
        except (
            TranscriptionCancelled,
            NoSpeechDetectedError,
            FileValidationError,
            OutputWriteError,
        ):
            raise
        except Exception as exc:
            if self._device_label.startswith("NVIDIA GPU") and self._looks_like_cuda_error(
                exc
            ):
                self._logger.warning(
                    "CUDA failed during transcription; retrying on CPU",
                    exc_info=True,
                )
                self._switch_to_cpu()
                status("CUDA 运行库不可用，正在切换到 CPU 并重新转写……")
                progress(0.0)
                return self.transcribe(
                    source,
                    destination,
                    cancel_event,
                    allow_download=False,
                    status_callback=status,
                    progress_callback=progress,
                )

            self._logger.exception("Transcription failed for %s", source)
            if self._looks_like_audio_error(exc):
                raise AudioDecodeError(
                    "无法解码音频，文件可能损坏或格式不受支持。"
                ) from exc
            raise TranscriptionError(
                "转写失败，详细信息已写入本地日志。"
            ) from exc

    def _load_model(
        self,
        allow_download: bool,
        status: StatusCallback,
    ) -> tuple[Any, str]:
        with self._model_lock:
            if self._model is not None:
                return self._model, self._device_label

            if not self.model_is_available() and not allow_download:
                raise ModelDownloadRequired(
                    "本地尚未安装语音识别模型，需要先下载模型。"
                )

            self.config.model_dir.mkdir(parents=True, exist_ok=True)
            candidates = self._device_candidates()
            factory = self._model_factory or self._default_model_factory()
            last_error: Exception | None = None

            for index, (device, compute_type) in enumerate(candidates):
                has_local_model = self.model_is_available()
                if has_local_model:
                    source = str(self.config.model_dir)
                    status(
                        "正在加载语音识别模型……"
                        if index == 0
                        else "GPU 不可用，正在切换到 CPU……"
                    )
                else:
                    source = self.config.model_name
                    status("正在下载并加载首次使用所需的模型……")

                kwargs: dict[str, Any] = {
                    "device": device,
                    "compute_type": compute_type,
                }
                if has_local_model:
                    kwargs["local_files_only"] = True
                else:
                    kwargs["download_root"] = str(self.config.model_dir)
                    kwargs["local_files_only"] = False

                try:
                    model = factory(source, **kwargs)
                except Exception as exc:
                    last_error = exc
                    self._logger.warning(
                        "Model load failed on %s/%s",
                        device,
                        compute_type,
                        exc_info=True,
                    )
                    if not has_local_model and not self.model_is_available():
                        raise ModelLoadError(
                            "模型下载失败，请检查网络、代理和磁盘空间。"
                        ) from exc
                    continue

                self._model = model
                self._device_label = (
                    "NVIDIA GPU (CUDA)" if device == "cuda" else f"CPU ({compute_type})"
                )
                self._logger.info(
                    "Speech model loaded: model=%s device=%s compute_type=%s",
                    self.config.model_name,
                    device,
                    compute_type,
                )
                return model, self._device_label

            raise ModelLoadError(
                "无法加载语音识别模型，请查看日志了解详细信息。"
            ) from last_error

    def _device_candidates(self) -> tuple[tuple[str, str], ...]:
        if self._configured_candidates is not None:
            if self._force_cpu:
                return tuple(
                    candidate
                    for candidate in self._configured_candidates
                    if candidate[0] == "cpu"
                )
            return self._configured_candidates

        candidates: list[tuple[str, str]] = []
        if not self._force_cpu:
            try:
                import ctranslate2

                if ctranslate2.get_cuda_device_count() > 0:
                    candidates.append(("cuda", "float16"))
            except Exception:
                self._logger.info("CUDA detection failed; using CPU", exc_info=True)

        candidates.extend((("cpu", "int8"), ("cpu", "float32")))
        return tuple(candidates)

    @staticmethod
    def _default_model_factory() -> ModelFactory:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ModelLoadError(
                "缺少 faster-whisper，请先安装 requirements.txt 中的依赖。"
            ) from exc
        return WhisperModel

    @staticmethod
    def _raise_if_cancelled(cancel_event: threading.Event) -> None:
        if cancel_event.is_set():
            raise TranscriptionCancelled("转写已取消，未生成或覆盖输出文件。")

    def _switch_to_cpu(self) -> None:
        with self._model_lock:
            self._model = None
            self._device_label = ""
            self._force_cpu = True
        gc.collect()

    @staticmethod
    def _looks_like_cuda_error(exc: Exception) -> bool:
        message = f"{type(exc).__name__}: {exc}".lower()
        markers = (
            "cublas",
            "cudnn",
            "cuda",
            "gpu",
            "out of memory",
        )
        return any(marker in message for marker in markers)

    @staticmethod
    def _looks_like_audio_error(exc: Exception) -> bool:
        message = f"{type(exc).__name__}: {exc}".lower()
        markers = (
            "audio",
            "decode",
            "averror",
            "invalid data",
            "ffmpeg",
            "format not found",
        )
        return any(marker in message for marker in markers)
