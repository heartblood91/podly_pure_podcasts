import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from app.logger import default_log_file, setup_logger


def test_file_handler_rotates_with_defaults(tmp_path: Path) -> None:
    logger = setup_logger("rotation-defaults", str(tmp_path / "app.log"))
    handler = next(h for h in logger.handlers if isinstance(h, RotatingFileHandler))
    assert handler.maxBytes == 10 * 1024 * 1024
    assert handler.backupCount == 5


def test_file_handler_rotates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODLY_LOG_MAX_BYTES", "2000")
    monkeypatch.setenv("PODLY_LOG_BACKUP_COUNT", "2")
    logger = setup_logger("rotation-small", str(tmp_path / "app.log"), logging.INFO)
    for index in range(200):
        logger.info("line %d %s", index, "x" * 50)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["app.log", "app.log.1", "app.log.2"]


def test_helper_processes_do_not_rotate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PODLY_LOG_ROTATE", "0")
    logger = setup_logger("rotation-off", str(tmp_path / "app.log"))
    assert not any(isinstance(h, RotatingFileHandler) for h in logger.handlers)


def test_log_file_is_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PODLY_LOG_FILE", raising=False)
    assert default_log_file() == "src/instance/logs/app.log"
    monkeypatch.setenv("PODLY_LOG_FILE", "src/instance/logs/writer.log")
    assert default_log_file() == "src/instance/logs/writer.log"
