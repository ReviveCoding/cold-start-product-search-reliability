"""Finalize pytest before native libraries can stall interpreter teardown."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


_EXIT_STATUS = 3


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del session
    global _EXIT_STATUS
    _EXIT_STATUS = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    del terminalreporter, config
    marker = os.getenv("PRODUCT_SEARCH_PYTEST_COMPLETION_MARKER")
    if marker:
        Path(marker).write_text(
            json.dumps({"exitstatus": int(exitstatus), "session_exitstatus": _EXIT_STATUS})
            + "\n",
            encoding="utf-8",
        )
    if os.getenv("PRODUCT_SEARCH_PYTEST_HARD_EXIT", "0") == "1":
        # Coverage and terminal-summary hooks have finished. Bypass rare XGBoost/OpenMP
        # interpreter-teardown stalls without changing the pytest exit status.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(int(exitstatus))
