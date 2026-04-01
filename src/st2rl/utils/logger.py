# -*- coding: utf-8 -*-
"""Logging utilities for STS2 RL."""

import logging
import os
from logging.handlers import RotatingFileHandler


def _default_log_level() -> int:
    value = str(os.getenv("ST2RL_LOG_LEVEL", "WARNING")).upper()
    return getattr(logging, value, logging.WARNING)


def setup_logger(name: str, log_file: str, level: int | None = None) -> logging.Logger:
    """Set up a logger with file and console handlers

    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level

    Returns:
        Configured logger
    """
    level = _default_log_level() if level is None else level

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers
    if not logger.handlers:
        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    else:
        for handler in logger.handlers:
            handler.setLevel(level)

    return logger


# Create loggers
run_logger = setup_logger(
    'run',
    'logs/run.log',
    logging.WARNING
)

train_logger = setup_logger(
    'train',
    'logs/train.log',
    logging.WARNING
)


def get_run_logger() -> logging.Logger:
    """Get the run logger"""
    return run_logger


def get_train_logger() -> logging.Logger:
    """Get the train logger"""
    return train_logger
