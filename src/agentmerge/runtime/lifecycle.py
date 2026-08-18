"""Process-lifecycle safeguards for long-running GPU experiments."""

from __future__ import annotations

import contextlib
import ctypes
import os
import signal
import sys
import threading
from typing import Dict, Iterator, Optional


_PR_SET_PDEATHSIG = 1


def arm_parent_death_signal(
    expected_parent_pid: int,
    death_signal: int = signal.SIGTERM,
) -> bool:
    """Ask Linux to signal this process if its current parent disappears.

    The parent PID is checked after ``prctl`` to close the race where the
    parent exits immediately before the kernel setting becomes active.
    """

    if not sys.platform.startswith("linux"):
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.prctl(
        _PR_SET_PDEATHSIG,
        int(death_signal),
        0,
        0,
        0,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number) if error_number else "prctl failed",
        )
    if os.getppid() != int(expected_parent_pid):
        os.kill(os.getpid(), int(death_signal))
    return True


def install_worker_termination_handlers() -> None:
    """Turn TERM/HUP into normal interpreter shutdown inside GPU workers."""

    def terminate(signum, _frame) -> None:
        # Avoid a second TERM interrupting Adapter/vLLM cleanup.
        signal.signal(signum, signal.SIG_IGN)
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, terminate)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, terminate)


@contextlib.contextmanager
def graceful_termination_signals(
    *, watch_parent: bool = False
) -> Iterator[None]:
    """Make TERM/HUP unwind Python ``finally`` blocks on the main thread."""

    if threading.current_thread() is not threading.main_thread():
        yield
        return

    previous = {}  # type: Dict[int, object]

    def terminate(signum, _frame) -> None:
        # Cleanup is allowed to finish; SIGKILL remains the force-stop path.
        signal.signal(signum, signal.SIG_IGN)
        raise SystemExit(128 + int(signum))

    handled = [signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled.append(signal.SIGHUP)
    for signum in handled:
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, terminate)

    try:
        if watch_parent:
            expected_parent_pid = int(
                os.getenv("AGENTMERGE_EXPECTED_PARENT_PID", str(os.getppid()))
            )
            arm_parent_death_signal(expected_parent_pid)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def parent_watch_enabled(value: Optional[str] = None) -> bool:
    raw = os.getenv("AGENTMERGE_WATCH_PARENT", "0") if value is None else value
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "arm_parent_death_signal",
    "graceful_termination_signals",
    "install_worker_termination_handlers",
    "parent_watch_enabled",
]
