"""Utility functions."""

import time

from loguru import logger


def timeit(func):
    """_summary_.

    Args:
        func: _description_

    """

    def wrapper(*args, **kwargs):
        """_summary_.

        Returns:
            _description_

        """
        ts = time.perf_counter()
        result = func(*args, **kwargs)
        te = time.perf_counter()
        total_time = te - ts
        logger.info(f"Function {func.__name__} took {total_time:.4f} seconds")
        return result

    return wrapper
