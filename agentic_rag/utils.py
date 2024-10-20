"""Utility functions."""

from typing import Any
import time
from collections import defaultdict

from loguru import logger


def timeit(func) -> Any:
    """Decorator for timing function.

    Args:
        func: Input function

    """

    def wrapper(*args, **kwargs) -> Any:
        """Wrapper to calculate time spent by function.

        Returns:
            Result from the function.

        """
        ts = time.perf_counter()
        result = func(*args, **kwargs)
        te = time.perf_counter()
        total_time = te - ts
        logger.info(f"Function {func.__name__} took {total_time:.4f} seconds")
        return result

    return wrapper


def rrf(list_of_texts: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion.

    For each list, we calculate a score using on RRF formula based
    on it's position in the list. We assign a score for each text
    in the lists.
    The original paper uses k=60 for best results:
    https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

    Args:
        list_of_texts: List of list of texts to be ranked.
            Each list is contains list of texts.
        k: Constant for smoothing. Defaults to 60.

    Returns:
        List of reranked text and corresponding score.
    """
    fused_results: dict[str, float] = defaultdict(float)

    for result_list in list_of_texts:
        for position, text in enumerate(result_list):
            fused_results[text] += 1.0 / (position + k)

    # Sort items based on their RRF scores in descending order
    sorted_items = dict(sorted(fused_results.items(), key=lambda x: x[1], reverse=True))
    return [(text, score) for text, score in sorted_items.items()]  # noqa: C416
