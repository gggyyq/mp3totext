from pathlib import Path

import pytest

from local_stt.file_io import (
    AtomicTextWriter,
    FileValidationError,
    default_output_path,
    normalize_output_path,
    validate_input_file,
    validate_output_path,
)


def test_default_output_path_handles_chinese_and_spaces(tmp_path: Path) -> None:
    audio = tmp_path / "会议 录音.mp3"
    assert default_output_path(audio) == tmp_path / "会议 录音.txt"


def test_normalize_output_path_adds_txt_extension(tmp_path: Path) -> None:
    assert normalize_output_path(tmp_path / "结果") == tmp_path / "结果.txt"


def test_validate_input_accepts_supported_audio(tmp_path: Path) -> None:
    audio = tmp_path / "测试.MP3"
    audio.write_bytes(b"not-real-audio")
    assert validate_input_file(audio) == audio


def test_validate_input_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileValidationError, match="不存在"):
        validate_input_file(tmp_path / "missing.mp3")


def test_validate_input_rejects_unsupported_extension(tmp_path: Path) -> None:
    audio = tmp_path / "notes.txt"
    audio.write_text("text", encoding="utf-8")
    with pytest.raises(FileValidationError, match="暂不支持"):
        validate_input_file(audio)


def test_validate_output_requires_existing_parent(tmp_path: Path) -> None:
    with pytest.raises(FileValidationError, match="输出目录不存在"):
        validate_output_path(tmp_path / "missing" / "result.txt")


def test_atomic_writer_commits_utf8_text(tmp_path: Path) -> None:
    output = tmp_path / "转写结果.txt"
    with AtomicTextWriter(output) as writer:
        assert writer.write_segment(" 你好，世界。 ") is True
        assert writer.write_segment("   ") is False
        writer.write_segment("第二行")
        writer.commit()

    assert output.read_text(encoding="utf-8") == "你好，世界。\n第二行\n"
    assert writer.line_count == 2
    assert writer.character_count == len("你好，世界。第二行")
    assert list(tmp_path.glob("*.part")) == []


def test_atomic_writer_failure_preserves_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing.txt"
    output.write_text("原内容", encoding="utf-8")

    with pytest.raises(RuntimeError):
        with AtomicTextWriter(output) as writer:
            writer.write_segment("不完整的新内容")
            raise RuntimeError("simulated failure")

    assert output.read_text(encoding="utf-8") == "原内容"
    assert list(tmp_path.glob("*.part")) == []
