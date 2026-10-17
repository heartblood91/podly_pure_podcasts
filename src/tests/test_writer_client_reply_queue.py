"""The writer client must not leak manager-owned reply queues in the writer.

Each Queue proxy unpickled by the manager server is pinned forever in
``Server.id_to_local_proxy_obj``. Before the fix every ``wait=True`` call
created a fresh ``manager.Queue()``, so the writer grew without bound.
"""

import multiprocessing
import socket
import threading
import time
import uuid
from typing import Any, Iterator, Tuple

import pytest

from app import ipc
from app.writer import client as client_module
from app.writer.client import WriterClient
from app.writer.protocol import WriteCommand, WriteCommandType, WriteResult

AUTHKEY = b"test-writer-client"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _proxy_counts() -> Tuple[int, int]:
    server = getattr(multiprocessing.current_process(), "_manager_server", None)
    if server is None:
        return (-1, -1)
    return len(server.id_to_local_proxy_obj), len(server.id_to_obj)


def _run_fake_writer(port: int) -> None:
    ipc.QueueManager.register("proxy_counts", callable=_proxy_counts)
    manager = ipc.make_server_manager(address=("127.0.0.1", port), authkey=AUTHKEY)
    server = manager.get_server()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    queue = ipc.get_queue()
    while True:
        cmd = queue.get()
        params = cmd.data.get("params", {}) if isinstance(cmd.data, dict) else {}
        delay = params.get("delay", 0)
        if delay:
            time.sleep(delay)
        if cmd.reply_queue:
            cmd.reply_queue.put(WriteResult(cmd.id, True, data={"echo": cmd.id}))


@pytest.fixture
def writer_port(monkeypatch: pytest.MonkeyPatch) -> Iterator[int]:
    port = _free_port()
    proc = multiprocessing.get_context("fork").Process(
        target=_run_fake_writer, args=(port,), daemon=True
    )
    proc.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.05)
    ipc.QueueManager.register("proxy_counts")
    monkeypatch.setattr(
        client_module,
        "make_client_manager",
        lambda: ipc.make_client_manager(address=("127.0.0.1", port), authkey=AUTHKEY),
    )
    yield port
    proc.kill()
    proc.join(5)


def _server_counts(client: WriterClient) -> Tuple[int, int]:
    return tuple(client.manager.proxy_counts()._getvalue())  # type: ignore[return-value]


def _action(client: WriterClient, **params: Any) -> Any:
    return client.action("dequeue_job", params, wait=True)


def test_reply_queue_proxies_stay_bounded(writer_port: int) -> None:
    client = WriterClient()
    for _ in range(3000):
        result = _action(client)
        assert result.success

    local_proxies, objects = _server_counts(client)
    # One reply queue for this thread (+ the command queue itself).
    assert local_proxies <= 2
    assert objects <= 3


def test_reply_queue_proxies_bounded_per_thread(writer_port: int) -> None:
    client = WriterClient()

    def worker() -> None:
        for _ in range(300):
            assert _action(client).success

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    local_proxies, _ = _server_counts(client)
    assert local_proxies <= 4 + 1


def test_fresh_queue_per_call_leaks(writer_port: int) -> None:
    """Documents the original behaviour that this client avoids."""
    client = WriterClient()
    client.connect()
    for _ in range(200):
        reply_q = client.manager.Queue()
        cmd = WriteCommand(
            id=str(uuid.uuid4()),
            type=WriteCommandType.ACTION,
            model=None,
            data={"action": "noop", "params": {}},
            reply_queue=reply_q,
        )
        client.queue.put(cmd)
        reply_q.get(timeout=5)

    local_proxies, _ = _server_counts(client)
    assert local_proxies >= 200


def test_stale_reply_after_timeout_is_discarded(writer_port: int) -> None:
    client = WriterClient()
    with pytest.raises(TimeoutError):
        client.submit(
            WriteCommand(
                id="slow",
                type=WriteCommandType.ACTION,
                model=None,
                data={"action": "noop", "params": {"delay": 0.5}},
            ),
            wait=True,
            timeout=0.1,
        )

    # The late "slow" reply lands in the same reused queue and must be skipped.
    result = _action(client)
    assert result.success
    assert result.data["echo"] == result.command_id
    assert result.command_id != "slow"


def test_reconnect_invalidates_thread_reply_queue(writer_port: int) -> None:
    client = WriterClient()
    assert _action(client).success
    first_queue = client._local.reply_queue  # pylint: disable=protected-access

    client._invalidate()  # pylint: disable=protected-access
    assert _action(client).success
    assert client._local.reply_queue is not first_queue  # pylint: disable=protected-access
