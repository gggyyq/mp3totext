from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from local_stt.config import AppConfig
from local_stt.transcriber import (
    LocalTranscriber,
    ModelDownloadRequired,
    NoSpeechDetectedError,
    TranscriptionCancelled,
)


@dataclass
class FakeSegment:
    text: str
    end: float


@dataclass
class FakeInfo:
    language: str = "zh"
    language_probability: float = 0.96
    duration: float = 10.0


class FakeModel:
    def __init__(self, segments: list[FakeSegment] | None = None) -> None:
        self.segments = segments or [
            FakeSegment(" 第一段。 ", 3.0),
            FakeSegment("第二段。", 10.0),
        ]
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def transcribe(self, audio_path: str, **kwargs: Any):
        self.calls.append((audio_path, kwargs))
        return iter(self.segments), FakeInfo()


class MissingCudaRuntimeModel:
    def transcribe(self, _audio_path: str, **_kwargs: Any):
        raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")


def make_cached_config(tmp_path: Path) -> AppConfig:
    config = AppConfig(data_dir=tmp_path / "应用数据")
    config.model_dir.mkdir(parents=True)
    for filename in ("model.bin", "config.json", "tokenizer.json"):
        (config.model_dir / filename).write_bytes(b"cached")
    return config


def make_audio(tmp_path: Path) -> Path:
    audio = tmp_path / "测试音频.mp3"
    audio.write_bytes(b"fake audio")
    return audio


def test_transcribe_uses_original_language_and_expected_options(
    tmp_path: Path,
) -> None:
    config = make_cached_config(tmp_path)
    fake_model = FakeModel()
    factory_calls: list[tuple[str, dict[str, Any]]] = []

    def factory(source: str, **kwargs: Any) -> FakeModel:
        factory_calls.append((source, kwargs))
        return fake_model

    transcriber = LocalTranscriber(
        config,
        model_factory=factory,
        device_candidates=(("cpu", "int8"),),
    )
    audio = make_audio(tmp_path)
    output = tmp_path / "转写.txt"
    progress: list[float] = []

    result = transcriber.transcribe(
        audio,
        output,
        threading.Event(),
        allow_download=False,
        progress_callback=progress.append,
    )

    assert factory_calls[0][0] == str(config.model_dir)
    assert factory_calls[0][1]["local_files_only"] is True
    _, options = fake_model.calls[0]
    assert options["task"] == "transcribe"
    assert options["language"] is None
    assert options["vad_filter"] is True
    assert options["word_timestamps"] is False
    assert output.read_text(encoding="utf-8") == "第一段。\n第二段。\n"
    assert result.language == "zh"
    assert result.character_count == len("第一段。第二段。")
    assert progress[-1] == 100.0


def test_missing_model_requires_download_approval(tmp_path: Path) -> None:
    config = AppConfig(data_dir=tmp_path / "应用数据")
    transcriber = LocalTranscriber(
        config,
        model_factory=lambda *_args, **_kwargs: FakeModel(),
        device_candidates=(("cpu", "int8"),),
    )

    with pytest.raises(ModelDownloadRequired, match="需要先下载"):
        transcriber.transcribe(
            make_audio(tmp_path),
            tmp_path / "result.txt",
            threading.Event(),
            allow_download=False,
        )


def test_no_speech_does_not_create_or_replace_output(tmp_path: Path) -> None:
    config = make_cached_config(tmp_path)
    output = tmp_path / "existing.txt"
    output.write_text("原内容", encoding="utf-8")
    fake_model = FakeModel(segments=[FakeSegment("   ", 10.0)])
    transcriber = LocalTranscriber(
        config,
        model_factory=lambda *_args, **_kwargs: fake_model,
        device_candidates=(("cpu", "int8"),),
    )

    with pytest.raises(NoSpeechDetectedError, match="未检测到"):
        transcriber.transcribe(
            make_audio(tmp_path),
            output,
            threading.Event(),
            allow_download=False,
        )

    assert output.read_text(encoding="utf-8") == "原内容"
    assert list(tmp_path.glob("*.part")) == []


def test_cancel_before_start_keeps_output_untouched(tmp_path: Path) -> None:
    config = make_cached_config(tmp_path)
    output = tmp_path / "existing.txt"
    output.write_text("原内容", encoding="utf-8")
    cancel_event = threading.Event()
    cancel_event.set()
    transcriber = LocalTranscriber(
        config,
        model_factory=lambda *_args, **_kwargs: FakeModel(),
        device_candidates=(("cpu", "int8"),),
    )

    with pytest.raises(TranscriptionCancelled):
        transcriber.transcribe(
            make_audio(tmp_path),
            output,
            cancel_event,
            allow_download=False,
        )

    assert output.read_text(encoding="utf-8") == "原内容"


def test_model_is_reused_between_tasks(tmp_path: Path) -> None:
    config = make_cached_config(tmp_path)
    fake_model = FakeModel()
    factory_call_count = 0

    def factory(_source: str, **_kwargs: Any) -> FakeModel:
        nonlocal factory_call_count
        factory_call_count += 1
        return fake_model

    transcriber = LocalTranscriber(
        config,
        model_factory=factory,
        device_candidates=(("cpu", "int8"),),
    )
    audio = make_audio(tmp_path)

    for index in range(2):
        transcriber.transcribe(
            audio,
            tmp_path / f"result-{index}.txt",
            threading.Event(),
            allow_download=False,
        )

    assert factory_call_count == 1


def test_cuda_runtime_failure_retries_on_cpu(tmp_path: Path) -> None:
    config = make_cached_config(tmp_path)
    cpu_model = FakeModel()
    devices: list[str] = []
    statuses: list[str] = []

    def factory(_source: str, **kwargs: Any):
        device = str(kwargs["device"])
        devices.append(device)
        if device == "cuda":
            return MissingCudaRuntimeModel()
        return cpu_model

    transcriber = LocalTranscriber(
        config,
        model_factory=factory,
        device_candidates=(("cuda", "float16"), ("cpu", "int8")),
    )
    output = tmp_path / "fallback.txt"

    result = transcriber.transcribe(
        make_audio(tmp_path),
        output,
        threading.Event(),
        allow_download=False,
        status_callback=statuses.append,
    )

    assert devices == ["cuda", "cpu"]
    assert result.device == "CPU (int8)"
    assert output.read_text(encoding="utf-8") == "第一段。\n第二段。\n"
    assert any("切换到 CPU" in status for status in statuses)
