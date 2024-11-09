"""Main application entrypoint."""

import click
from loguru import logger
from typing import Optional
from agentic_rag.configs.config import COLLECTION_NAME, TOP_K, DEFAULT_QUESTION
from agentic_rag.configs.prompts import RAG_PROMPT
from agentic_rag.data_pipeline.data_pipeline import build_data_pipeline
from agentic_rag.data_pipeline.indexing import build_index
from agentic_rag.retrieval.retrieval_engine import RetrievalEngine

logger.add("rag_agent.log")


@click.command(
    help="""
Run pipelines.

Examples:
  \n
  # Run the data pipeline\n
    python agentic_rag/run.py data

  # Run the data pipeline\n
    python agentic_rag/run.py query --query-text "My question?"
"""
)
@click.argument(
    "pipeline",
    type=click.Choice(["data", "query"]),
    required=True,
)
@click.option(
    "--query-text",
    "query_text",
    is_flag=False,
    default=DEFAULT_QUESTION,
    help="Input question.",
)
def main(pipeline: str, query_text: Optional[str] = None) -> None:
    """Run data or query pipelines.

    Args:
        pipeline: Name of the pipeline to run
        query_text: Input query. Defaults to None.

    Raises:
        click.UsageError: If `query_text` is not passed when
            pipeline is in query mode.
    """
    if pipeline == "data":
        logger.info("Running data pipeline...")
        ds = build_data_pipeline()

        logger.info("Building index...")
        build_index(ds)

    if pipeline == "query":
        logger.info(f"Running query : {query_text}")
        retrieval_engine = RetrievalEngine(
            query=query_text,
            collection_name=COLLECTION_NAME,
            top_k=TOP_K,
        )
        llm_response = retrieval_engine.user_plan(prompt_template=RAG_PROMPT)
        logger.info("-" * 100)
        logger.info(f"Query:\n{query_text}\nLLM Response:\n{llm_response}")


if __name__ == "__main__":
    main()
