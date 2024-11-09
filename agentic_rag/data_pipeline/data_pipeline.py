"""Datapipeline function."""

# Warning related to star imports
# ruff: noqa: F403 F405
from functools import partial

import ray
from loguru import logger
from ray.data import Dataset

from agentic_rag.configs.config import *
from agentic_rag.data_preprocess.data import chunk_text, parse_html
from agentic_rag.data_preprocess.embed import (
    EmbedChunks,
    FullTextEmbedChunks,
    SparseEmbedChunks,
)
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
def build_data_pipeline() -> Dataset:
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
    # 1. Get all html filenames
    ds = ray.data.from_items(
        [{"path": path} for path in DATA_DIR.rglob("*.html") if not path.is_dir()]
    )
    logger.info(f"Found {ds.count()} documents")

    # 2. Extract text from html
    content_ds = ds.flat_map(parse_html)

    # 3. Create chunks from text
    chunks_ds = content_ds.flat_map(
        partial(chunk_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    )

    # 4. Get dense embeddings for the chunks
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

    # 5. Get sparse embeddings for the chunks
    if ENABLE_SPARSE_INDEX:
        embedded_chunks = embedded_chunks.map_batches(
            SparseEmbedChunks,
            fn_constructor_kwargs={"batch_size": BATCH_SIZE},
            batch_size=BATCH_SIZE,
            num_gpus=0.5,  #  share 1 GPU with 2 models
            concurrency=1,
        )

    # 6. Get sparse embedding using BM-25 model for the chunks
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
