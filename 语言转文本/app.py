from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import tkinter as tk

from local_stt.config import AppConfig
from local_stt.transcriber import LocalTranscriber
from local_stt.ui import SpeechToTextApp


def configure_logging(config: AppConfig) -> None:
    config.log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    file_handler = RotatingFileHandler(
        config.log_file,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=(file_handler, console_handler),
    )


def main() -> None:
    bundled_model = Path(__file__).resolve().parent / "whisper-large-v3-ct2"
    config = AppConfig(model_path=bundled_model if bundled_model.is_dir() else None)
    configure_logging(config)
    logging.getLogger(__name__).info("Application starting")

    root = tk.Tk()
    transcriber = LocalTranscriber(config)
    app = SpeechToTextApp(root, config, transcriber)
    root.mainloop()
    del app


if __name__ == "__main__":
    main()
