from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TextIO

from .config import SUPPORTED_AUDIO_EXTENSIONS


class FileValidationError(ValueError):
    """The selected input or output path is invalid."""


class OutputWriteError(OSError):
    """The transcript could not be safely written."""


def default_output_path(input_path: str | Path) -> Path:
    path = Path(input_path).expanduser()
    return path.with_suffix(".txt")


def normalize_output_path(output_path: str | Path) -> Path:
    path = Path(output_path).expanduser()
    if not path.suffix:
        path = path.with_suffix(".txt")
    return path


def validate_input_file(
    input_path: str | Path,
    supported_extensions: tuple[str, ...] = SUPPORTED_AUDIO_EXTENSIONS,
) -> Path:
    path = Path(input_path).expanduser()
    if not str(path).strip():
        raise FileValidationError("请选择音频文件。")
    if not path.exists() or not path.is_file():
        raise FileValidationError("所选音频文件不存在。")
    if path.suffix.lower() not in supported_extensions:
        supported = "、".join(supported_extensions)
        raise FileValidationError(f"暂不支持该音频格式。支持：{supported}")
    if not os.access(path, os.R_OK):
        raise FileValidationError("无法读取所选音频文件。")
    return path


def validate_output_path(output_path: str | Path) -> Path:
    raw_path = str(output_path).strip()
    if not raw_path:
        raise FileValidationError("请选择转写文本的保存位置。")

    path = normalize_output_path(raw_path)
    parent = path.parent
    if not parent.exists() or not parent.is_dir():
        raise FileValidationError("输出目录不存在，请重新选择保存位置。")
    if not os.access(parent, os.W_OK):
        raise FileValidationError("无法写入输出目录，请更换保存位置。")
    return path


class AtomicTextWriter:
    """Write a transcript to a temporary file and replace the target on success."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.temp_path: Path | None = None
        self._file: TextIO | None = None
        self._committed = False
        self.line_count = 0
        self.character_count = 0

    def __enter__(self) -> "AtomicTextWriter":
        try:
            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{self.output_path.name}.",
                suffix=".part",
                dir=self.output_path.parent,
                delete=False,
            )
        except OSError as exc:
            raise OutputWriteError("无法在输出目录创建临时文件。") from exc

        self._file = temp_file
        self.temp_path = Path(temp_file.name)
        return self

    def write_segment(self, text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return False
        if self._file is None:
            raise RuntimeError("文本写入器尚未打开。")

        try:
            self._file.write(cleaned)
            self._file.write("\n")
        except OSError as exc:
            raise OutputWriteError("写入转写文本失败，请检查磁盘空间和目录权限。") from exc

        self.line_count += 1
        self.character_count += len(cleaned)
        return True

    def commit(self) -> None:
        if self._file is None or self.temp_path is None:
            raise RuntimeError("文本写入器尚未打开。")

        try:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._file = None
            os.replace(self.temp_path, self.output_path)
            self._committed = True
        except OSError as exc:
            raise OutputWriteError(
                "无法保存转写文本，请确认文件未被其他程序占用。"
            ) from exc

    def abort(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None

        if self.temp_path is not None and self.temp_path.exists():
            try:
                self.temp_path.unlink()
            except OSError:
                pass

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._committed:
            self.abort()
