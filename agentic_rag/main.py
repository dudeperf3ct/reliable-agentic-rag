"""Main application entrypoint."""

from functools import partial
import ray
from loguru import logger
from ray.data import Dataset

from agentic_rag.configs.config import *
from agentic_rag.data_preprocess.data import chunk_text, parse_html
from agentic_rag.data_preprocess.embed import (
    EmbedChunks,
    SparseEmbedChunks,
    FullTextEmbedChunks,
)
from agentic_rag.vector_store.milvus import CustomMilvusClient
from agentic_rag.utils import timeit


@timeit
def get_corpus(chunk_ds: Dataset) -> list[str]:
    """Get a corpus of text to fit BM-25 model.

    Args:
        chunk_ds: Chunked dataset containing text and
            source.

    Returns:
        List of chunked text
    """
    content = chunk_ds.take_all()
    return [data["text"] for data in content]


@timeit
def data_pipeline() -> Dataset:
    """Run data pipeline using Ray Data.

    In this pipeline,
    1. We get all the html files for the docs dataset
    2. Parse the html files to extract content
    3. Chunk the content
    4. Get dense embeddings for the chunked text
    5. Optionally, add sparse embedding for chunked text
    6. Optionally, add full text sparse embedding using BM-25 model.

    Returns:
        Ray dataset containing text, source and dense embeddings.
        Sparse and full text embeddings are also returned if included.
    """
    # Get all html filenames
    ds = ray.data.from_items(
        [{"path": path} for path in DATA_DIR.rglob("*.html") if not path.is_dir()]
    )
    logger.info(f"Found {ds.count()} documents")

    # Extract text from html
    content_ds = ds.flat_map(parse_html)

    # Create chunks from text
    chunks_ds = content_ds.flat_map(
        partial(chunk_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    )

    # Get dense embeddings for the chunks
    embedded_chunks = chunks_ds.map_batches(
        EmbedChunks,
        fn_constructor_kwargs={
            "model_name": EMBEDDING_MODEL_NAME,
            "batch_size": BATCH_SIZE,
        },
        batch_size=BATCH_SIZE,
        num_gpus=0.5 if ENABLE_SPARSE_INDEX else 1,  # share 1 GPU with 2 models
        concurrency=1,
    )

    # Get sparse embeddings for the chunks
    if ENABLE_SPARSE_INDEX:
        embedded_chunks = embedded_chunks.map_batches(
            SparseEmbedChunks,
            fn_constructor_kwargs={"batch_size": BATCH_SIZE},
            batch_size=BATCH_SIZE,
            num_gpus=0.5,  #  share 1 GPU with 2 models
            concurrency=1,
        )

    # Get sparse embedding using BM-25 model for the chunks
    if ENABLE_FULL_TEXT_INDEX:
        corpus = get_corpus(chunks_ds)
        embedded_chunks = embedded_chunks.map_batches(
            FullTextEmbedChunks,
            fn_constructor_kwargs={"corpus": corpus},
            batch_size=BATCH_SIZE,
            num_cpus=4,
            concurrency=1,
        )
    return embedded_chunks


@timeit
def build_index(ds: Dataset) -> None:
    """Insert the data into milvus vector store.

    Args:
        ds: Ray dataset containing text, source and dense embeddings.
            Sparse and full text embeddings are also returned if included.
    """
    data = ds.take_all()
    logger.info(f"Before filtering length of data {len(data)}")
    # Remove empty full_text_vector fields
    # Example : https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/docs/cli.html#id1
    # This creates {'text': '-?#',....,'full_text_vector': {}}
    filtered_data = [item for item in data if item.get("full_text_vector")]
    logger.info(f"After filtering length of data {len(filtered_data)}")

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


if __name__ == "__main__":
    logger.info("Running data pipeline...")
    ds = data_pipeline()
    build_index(ds)
