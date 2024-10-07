from agentic_rag.configs.config import DATA_DIR
from agentic_rag.data_pipeline.data_pipeline import data_pipeline
from loguru import logger

if __name__ == "__main__":
    logger.info("Running data pipeline...")
    data_pipeline(DATA_DIR)
