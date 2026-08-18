"""
Logging for Simulacrum.

Every module does ``logger = get_logger(__name__)``, which produces names like
``simulacrum.solvers.nonlinear``. The whole job of this module is to make sure
that produces exactly one line of output per message.

Why the handler goes on the package logger and nowhere else
-----------------------------------------------------------
Python's loggers form a tree and a record is passed to the handlers of every
ancestor, not just its own. So attaching a handler to each module's logger --
the obvious thing, and what this module used to do -- means a record from
``simulacrum.solvers.nonlinear`` is emitted by that logger's handler and then
again by any handler on ``simulacrum`` or on the root. Two measured symptoms:

* ``get_logger("simulacrum")`` followed by ``get_logger("simulacrum.solvers")``
  gave two lines for one ``info()`` call;
* worse, any application that calls ``logging.basicConfig()`` -- which is most
  of them -- puts a handler on the *root* logger, and from then on every
  Simulacrum message appeared twice, in two different formats.

So the handler is attached once, to the top-level package logger, and the
per-module loggers carry none. They propagate up to it, which is exactly the
mechanism the tree exists for.

``propagate`` is then turned off on the package logger, which stops records
escaping into the application's root handler and being printed a second time.
That is a deliberate trade: it costs an application the ability to capture
Simulacrum's output through its own root handler. If you want that -- routing
to a file, or into pytest's ``caplog`` -- call :func:`use_application_logging`,
which removes the handler and restores propagation, and then configure the
``simulacrum`` logger however you like.

Levels
------
``get_logger`` sets a level only when one is passed. It used to set ``INFO``
unconditionally, so a later ``get_logger(name)`` from anywhere in the library
silently undid a caller's ``get_logger(name, logging.DEBUG)``. The package
logger is initialised at ``INFO`` once; module loggers are left at ``NOTSET``
so they inherit it, and :func:`set_level` changes it for everything at once.
"""

from __future__ import annotations

import contextlib
import logging

__all__ = ["get_logger", "set_level", "quiet", "use_application_logging",
           "PACKAGE"]

#: The logger that owns the handler. Everything else propagates to it.
PACKAGE = "simulacrum"

_FORMAT = "[%(name)s %(levelname)s] %(message)s"


def _package_of(name: str) -> str:
    """The top-level logger name, so this works for callers outside the
    package too (tests, notebooks)."""
    return name.split(".", 1)[0] or PACKAGE


def get_logger(name: str = PACKAGE, level: int | None = None
               ) -> logging.Logger:
    """The logger for ``name``, with the package handler installed once.

    Parameters
    ----------
    name : usually ``__name__``.
    level : applied to *this* logger if given. Left alone otherwise, so one
        module cannot reset another's verbosity as a side effect of importing.
        Use :func:`set_level` to change the package-wide level.
    """
    logger = logging.getLogger(name)
    root = logging.getLogger(_package_of(name))

    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        # stop records reaching the application's root handler as well; see
        # use_application_logging() to give that back
        root.propagate = False

    if level is not None:
        logger.setLevel(level)
    return logger


def set_level(level: int, name: str = PACKAGE) -> None:
    """Set the level for a whole subtree at once.

    Module loggers are left at ``NOTSET`` deliberately, so they inherit from
    the package logger and one call here changes all of them.
    """
    logging.getLogger(name).setLevel(level)


@contextlib.contextmanager
def quiet(level: int = logging.WARNING, name: str = PACKAGE):
    """Silence the library for the duration of a block, then restore it.

    The right way to stop a long solve from filling a terminal:

        with quiet():
            result = driver.run()

    **Do not reach for ``logging.disable`` instead.** It is process-global,
    it silences every library in the interpreter rather than this one, and it
    is not scoped to anything -- so a function that calls it leaves logging dead
    for everything that runs afterwards. Four examples in this repository did
    exactly that, and because one of them is imported by a test, it disabled
    logging for the remainder of every pytest session; the symptom was two
    unrelated tests in this module failing only when run as part of the full
    suite.
    """
    logger = logging.getLogger(name)
    previous = logger.level
    logger.setLevel(level)
    try:
        yield logger
    finally:
        logger.setLevel(previous)


def use_application_logging(name: str = PACKAGE) -> None:
    """Hand logging configuration back to the application.

    Removes the package handler and restores propagation, so records reach
    whatever the application configured -- ``basicConfig``, a file handler, or
    pytest's ``caplog``, which cannot see records that do not propagate.
    Idempotent.
    """
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.propagate = True
