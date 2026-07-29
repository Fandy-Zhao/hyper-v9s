from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

from .types import ComposeSelection


_CURRENT_SELECTION = ContextVar("compose_selection", default=None)


def get_current_selection() -> Optional[ComposeSelection]:
    return _CURRENT_SELECTION.get()


@contextmanager
def use_selection(selection: ComposeSelection) -> Iterator[None]:
    """Apply a selection to every ComposeLinear called in this context."""

    token = _CURRENT_SELECTION.set(selection)
    try:
        yield
    finally:
        _CURRENT_SELECTION.reset(token)
