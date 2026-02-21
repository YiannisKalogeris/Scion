"""Lightweight logging for Simulacrum."""

import logging


def get_logger(name: str = "simulacrum", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(name)s %(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
