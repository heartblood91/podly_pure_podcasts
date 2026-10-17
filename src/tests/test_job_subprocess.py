"""Per-episode jobs run in a short-lived child process."""

import contextlib
import sys
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from app import job_runner
from app import jobs_manager as jm


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> Any:
    fake_scheduler = MagicMock()
    fake_scheduler.app.app_context.side_effect = contextlib.nullcontext
    monkeypatch.setattr(jm, "scheduler", fake_scheduler)
    monkeypatch.setattr(jm, "_remove_session", lambda: None)
    failures: List[tuple] = []
    monkeypatch.setattr(
        jm,
        "mark_job_failed",
        lambda job_id, message, _status: failures.append((job_id, message)),
    )
    instance = jm.JobsManager.__new__(jm.JobsManager)
    instance._status_manager = MagicMock()  # pylint: disable=protected-access
    instance.failures = failures
    return instance


def _child(monkeypatch: pytest.MonkeyPatch, code: str) -> None:
    monkeypatch.setattr(
        jm,
        "job_runner_command",
        lambda job_id, post_guid: [sys.executable, "-c", code, job_id, post_guid],
    )


def test_subprocess_is_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PODLY_JOB_SUBPROCESS", raising=False)
    assert jm.job_subprocess_enabled()
    monkeypatch.setenv("PODLY_JOB_SUBPROCESS", "0")
    assert not jm.job_subprocess_enabled()


def test_successful_child_leaves_job_alone(
    manager: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PODLY_JOB_SUBPROCESS", raising=False)
    _child(monkeypatch, "import sys; sys.exit(0)")
    manager._process_job("job-1", "guid-1")  # pylint: disable=protected-access
    assert manager.failures == []


def test_crashed_child_fails_job(manager: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PODLY_JOB_SUBPROCESS", raising=False)
    _child(monkeypatch, "import os, signal; os.kill(os.getpid(), signal.SIGKILL)")
    manager._process_job("job-2", "guid-2")  # pylint: disable=protected-access
    assert manager.failures == [("job-2", "Job process exited with code -9")]


def test_hung_child_is_killed_and_job_failed(
    manager: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PODLY_JOB_TIMEOUT_SECONDS", "0.5")
    _child(monkeypatch, "import time; time.sleep(30)")
    manager._process_job("job-3", "guid-3")  # pylint: disable=protected-access
    assert manager.failures == [("job-3", "Job process timed out after 0.5s")]


def test_child_receives_job_arguments(
    manager: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "args"
    _child(
        monkeypatch,
        f"import sys; open({str(out)!r}, 'w').write(' '.join(sys.argv[1:]))",
    )
    manager._process_job("job-4", "guid-4")  # pylint: disable=protected-access
    assert out.read_text() == "job-4 guid-4"


def test_in_process_mode_runs_job_directly(
    manager: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PODLY_JOB_SUBPROCESS", "0")
    calls: List[tuple] = []
    monkeypatch.setattr(jm, "run_job", lambda *args: calls.append(args[:2]))
    manager._process_job("job-5", "guid-5")  # pylint: disable=protected-access
    assert calls == [("job-5", "guid-5")]


def test_child_env_points_at_src_and_job_log(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PODLY_JOB_LOG_FILE", raising=False)
    monkeypatch.setenv("PODLY_LOG_FILE", "/data/logs/app.log")
    monkeypatch.setenv("PYTHONPATH", "/extra")
    env = jm.job_runner_env()
    src_dir = str(Path(jm.__file__).resolve().parents[1])
    assert env["PYTHONPATH"].split(":") == [src_dir, "/extra"]
    assert env["PODLY_LOG_FILE"] == "/data/logs/jobs.log"
    assert jm.job_runner_command("j", "g")[1:] == ["-m", "app.job_runner", "j", "g"]


def test_job_runner_main(monkeypatch: pytest.MonkeyPatch) -> None:
    import app as app_package
    from app import jobs_manager

    fake_app = MagicMock()
    fake_app.app_context.side_effect = contextlib.nullcontext
    monkeypatch.setattr(app_package, "create_job_runner_app", lambda: fake_app)
    calls: List[tuple] = []
    monkeypatch.setattr(jobs_manager, "run_job", lambda *args: calls.append(args[:2]))

    assert job_runner.main(["job-6", "guid-6"]) == 0
    assert calls == [("job-6", "guid-6")]
    assert job_runner.main(["only-one"]) == 2
