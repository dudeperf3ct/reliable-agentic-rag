"""Build index."""

# Warning related to star imports
# ruff: noqa: F403 F405
from loguru import logger
from ray.data import Dataset
from agentic_rag.configs.config import *

from agentic_rag.utils import timeit
from agentic_rag.vector_store.milvus import CustomMilvusClient


@timeit
def build_index(ds: Dataset) -> None:
    """Insert the data into milvus vector store.

    In data indexing,
    1. Connect to vector database
    2. Create a collection
    3. Store data into the collection

    Args:
        ds: Ray dataset containing text, source and dense embeddings.
            Sparse and full text embeddings are also returned if included.

    """
    data = ds.take_all()

    logger.info(f"Length of data before filtering {len(data)}")
    # Remove empty full_text_vector fields
    # Example : https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/docs/cli.html#id1
    # This creates {'text': '-?#',....,'full_text_vector': {}}
    filtered_data = [item for item in data if item.get("full_text_vector")]
    logger.info(f"Length of data after filtering {len(filtered_data)}")

    milvus_client = CustomMilvusClient()
    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        dense_dim=EMBEDDING_MODEL_DIM,
        dense_distance_metric=DENSE_METRIC,
        add_sparse_index=ENABLE_SPARSE_INDEX,
        add_full_text_index=ENABLE_FULL_TEXT_INDEX,
    )
    milvus_client.store(filtered_data, COLLECTION_NAME)
    logger.info("Inserted data into milvus store successfully")
