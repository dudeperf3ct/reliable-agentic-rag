"""Application configuration."""

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = ROOT_DIR / "dataset"

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
