from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_data_dir


SUPPORTED_AUDIO_EXTENSIONS = (
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
    ".webm",
)


def _default_data_dir() -> Path:
    return Path(
        user_data_dir(
            appname="LocalSpeechTranscriber",
            appauthor=False,
            roaming=False,
        )
    )


@dataclass(frozen=True, slots=True)
class AppConfig:
    app_name: str = "本地音频转文字"
    model_name: str = "large-v3"
    model_path: Path | None = None
    beam_size: int = 5
    vad_filter: bool = True
    data_dir: Path = field(default_factory=_default_data_dir)
    supported_extensions: tuple[str, ...] = SUPPORTED_AUDIO_EXTENSIONS

    @property
    def model_dir(self) -> Path:
        if self.model_path is not None:
            return self.model_path
        return self.data_dir / "models" / self.model_name

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def log_file(self) -> Path:
        return self.log_dir / "app.log"
