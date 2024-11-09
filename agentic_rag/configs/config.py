"""Application configuration."""

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = ROOT_DIR / "dataset"

# Data pipeline parameters
BATCH_SIZE = 128
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_DIM = 384
DENSE_METRIC = "COSINE"
SPARSE_METRIC = "IP"
ENABLE_SPARSE_INDEX = False
ENABLE_FULL_TEXT_INDEX = True
COLLECTION_NAME = "nvidiatritondocs"

# Retrieval parameters
TOP_K = 5
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RETRIEVAL_PRIORITY = [
    "no_retrieval",
    "semantic_search",
    "hybrid_search_with_rrf",
    "hybrid_search_with_reranker",
    "hyde_retrieval",
]
TRUST_SCORE_THRESH = 0.98
STUB_RESPONSE = "This question cannot be handled without additional clarification or further information."
DEBUG_WITHOUT_LLM = False

## LLM parameters
LLM_MODEL = "claude-3-haiku-20240307"
LLM_API_BASE = None
MAX_OUTPUT_TOKENS = 500
DEFAULT_QUESTION = "How to make custom layers of TensorRT work in Triton?"
