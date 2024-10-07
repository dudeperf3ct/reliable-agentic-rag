"""Data preprocessing pipeline."""

from loguru import logger
import ray
from agentic_rag.data_pipeline.dataset import parse_html
from agentic_rag.utils import timeit


@timeit
def data_pipeline(data_dir: str):
    """_summary_

    Args:
        data_dir: _description_
    """
    ds = ray.data.from_items(
        [{"path": path} for path in data_dir.rglob("*.html") if not path.is_dir()]
    )
    logger.info(f"Found {ds.count()} documents")
    content_ds = ds.flat_map(parse_html)
    logger.info(content_ds.take(1))
