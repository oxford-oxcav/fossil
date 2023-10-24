# Copyright (c) 2023, Alessandro Abate, Alec Edwards, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

import logging


class Logger:
    _loggers = {}
    log_level = logging.WARNING

    @classmethod
    def set_logger_level(cls, verbosity: int):
        """Set the logger level."""
        if verbosity == 0:
            cls.log_level = logging.WARNING
        elif verbosity == 1:
            cls.log_level = logging.INFO
        elif verbosity == 2:
            cls.log_level = logging.DEBUG

        for logger in cls._loggers.values():
            logger.setLevel(cls.log_level)

    @classmethod
    def setup_logger(cls, name, verbosity: int = 0):
        """Return a logger with the given name."""
        logger = logging.getLogger(name)

        logger.setLevel(cls.log_level)  # Set the desired logging level

        # Create a console handler
        ch = logging.StreamHandler()
        # ch.setLevel(cls.log_level)

        # Create a formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Add formatter to the console handler
        ch.setFormatter(formatter)

        # Add the console handler to the logger
        logger.addHandler(ch)

        cls._loggers[name] = logger
        return logger
