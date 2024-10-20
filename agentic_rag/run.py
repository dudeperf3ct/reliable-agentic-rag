"""Main application entrypoint."""

import click
from loguru import logger

from agentic_rag.configs.config import COLLECTION_NAME, TOP_K
from agentic_rag.data_pipeline.data_pipeline import build_data_pipeline
from agentic_rag.data_pipeline.indexing import build_index
from agentic_rag.retrieval.retrieval_engine import RetrievalEngine


@click.command(
    help="""
Run RAG workflow.

Examples:
  \b
  # Run the data pipeline
    python run.py --datapipline
"""
)
@click.option("--data-pipeline", is_flag=True, default=False, help="Run datapipeline")
def main(data_pipeline: bool = False):
    if data_pipeline:
        logger.info("Running data pipeline...")
        ds = build_data_pipeline()

        logger.info("Building index...")
        build_index(ds)

    for q in ["TensorRT", "PyTorch", "ONNX"]:
        query = f"How to make custom layers of {q} work in Triton?"
        logger.info(f"Query: {query}")
        retrieval_engine = RetrievalEngine(
            query=query,
            collection_name=COLLECTION_NAME,
            top_k=TOP_K,
        )
        retrieval_engine.user_plan()


if __name__ == "__main__":
    main()
