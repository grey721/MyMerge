import os
import signal

import pytest

from agentmerge.runtime.lifecycle import (
    graceful_termination_signals,
    parent_watch_enabled,
)


def test_parent_watch_flag_is_explicit():
    assert parent_watch_enabled("true") is True
    assert parent_watch_enabled("1") is True
    assert parent_watch_enabled("false") is False
    assert parent_watch_enabled("") is False


def test_sigterm_becomes_clean_system_exit_and_handler_is_restored():
    previous = signal.getsignal(signal.SIGTERM)

    with pytest.raises(SystemExit) as raised:
        with graceful_termination_signals():
            os.kill(os.getpid(), signal.SIGTERM)

    assert raised.value.code == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) is previous
