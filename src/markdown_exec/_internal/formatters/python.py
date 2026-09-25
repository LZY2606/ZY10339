# Formatter for executing Python code.

from __future__ import annotations

import re
import signal
import sys
import traceback
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from io import StringIO
from types import ModuleType
from typing import TYPE_CHECKING, Any

from markdown_exec._internal.formatters._exec_python import exec_python
from markdown_exec._internal.formatters.base import ExecutionError, base_format
from markdown_exec._internal.rendering import code_block

if TYPE_CHECKING:
    from collections.abc import Iterator

_sessions_globals: dict[str, dict] = defaultdict(dict)
_sessions_counter: dict[str | None, int] = defaultdict(int)
_code_blocks: dict[str, list[str]] = {}


class _ExecutionTimeoutError(Exception):
    """Exception raised when a code block exceeds its time limit."""


def _raise_timeout(signum: int, frame: Any) -> None:  # noqa: ARG001
    raise _ExecutionTimeoutError


@contextmanager
def _time_limit(seconds: float) -> Iterator[None]:
    try:
        previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    except (ValueError, OSError, AttributeError):
        # Not in the main thread, or signals unsupported: cannot enforce the timeout.
        yield
        return
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _buffer_print(buffer: StringIO, *texts: str, end: str = "\n", **kwargs: Any) -> None:  # noqa: ARG001
    buffer.write(" ".join(str(text) for text in texts) + end)


def _code_block_id(
    id: str | None = None,  # noqa: A002
    session: str | None = None,
    title: str | None = None,
) -> str:
    _sessions_counter[session] += 1
    if id:
        code_block_id = f"id {id}"
    elif session:
        code_block_id = f"session {session}; n{_sessions_counter[session]}"
        if title:
            code_block_id = f"{code_block_id}; title {title}"
    else:
        code_block_id = f"n{_sessions_counter[session]}"
        if title:
            code_block_id = f"{code_block_id}; title {title}"
    return f"<code block: {code_block_id}>"


def _run_python(
    code: str,
    returncode: int | None = None,  # noqa: ARG001
    session: str | None = None,
    id: str | None = None,  # noqa: A002
    *,
    session_store: dict | None = None,
    timeout: float | None = None,
    **extra: str,
) -> str:
    title = extra.get("title")
    code_block_id = _code_block_id(id, session, title)
    _code_blocks[code_block_id] = code.split("\n")
    if session:
        store = session_store if session_store is not None else _sessions_globals
        exec_globals = store.setdefault(session, {})
    else:
        exec_globals = {}

    # Other libraries expect functions to have a valid `__module__` attribute.
    # To achieve this, we need to add a `__name__` attribute to the globals.
    # We compute the name from the code block ID, replacing invalid characters with `_`.
    # We also create a module object with the same name and add it to `sys.modules`,
    # because that's what yet other libraries expect (`dataclasses` for example).
    module_name = re.sub(r"[^a-zA-Z\d]+", "_", code_block_id)
    exec_globals["__name__"] = module_name
    sys.modules[module_name] = ModuleType(module_name)

    buffer = StringIO()
    exec_globals["print"] = partial(_buffer_print, buffer)

    # Snapshot the session state so that a timed-out block
    # cannot leak partial state into the next code block.
    snapshot = dict(exec_globals) if timeout else None
    try:
        if timeout:
            with _time_limit(timeout):
                exec_python(code, code_block_id, exec_globals)
        else:
            exec_python(code, code_block_id, exec_globals)
    except _ExecutionTimeoutError:
        if snapshot is not None:
            exec_globals.clear()
            exec_globals.update(snapshot)
        raise ExecutionError(
            code_block("python", f"Execution timed out after {timeout} seconds.", **extra),
        ) from None
    except Exception as error:
        trace = traceback.TracebackException.from_exception(error)
        for frame in trace.stack:
            if frame.filename.startswith("<code block: "):
                if sys.version_info >= (3, 13):
                    frame._lines = _code_blocks[frame.filename][frame.lineno - 1]  # ty:ignore[unresolved-attribute,unsupported-operator,unused-ignore-comment,unused-ignore-comment]
                else:
                    frame._line = _code_blocks[frame.filename][frame.lineno - 1]  # ty:ignore[unresolved-attribute,unsupported-operator,unused-ignore-comment,unused-ignore-comment]
        raise ExecutionError(code_block("python", "".join(trace.format()), **extra)) from error
    return buffer.getvalue()


def _format_python(**kwargs: Any) -> str:
    return base_format(language="python", run=_run_python, **kwargs)
