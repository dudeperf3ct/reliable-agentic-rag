"""Embedding modules."""

from pathlib import Path
from typing import Any, Optional

import torch
from loguru import logger
from milvus_model.hybrid import BGEM3EmbeddingFunction
from milvus_model.sparse import BM25EmbeddingFunction
from scipy.sparse import csr_array
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sparse_to_dict(sparse_array: csr_array) -> dict[int, float]:
    """
    Convert sparse array in csr format to a dictionary.

    Args:
        sparse_array: Sparse array in csr format

    Returns:
        Dictionary mapping tokens to score.

    """
    _, col_indices = sparse_array.nonzero()
    non_zero_values = sparse_array.data
    result_dict = {}
    for col_index, value in zip(col_indices, non_zero_values):
        result_dict[int(col_index)] = float(value)
    return result_dict


class EmbedChunks:
    """
    Embedding class using SentenceTransformer library.

    Attributes:
        embedding_model: Embedding model loaded using
            sentence transformer library.
        batch_size: Batch size

    """

    def __init__(self, model_name: str, batch_size: int = 128) -> None:
        """
        Init function for EmbedChunks class.

        Args:
            model_name: Model name compatible with sentence
                transformers library.
            batch_size: Batch size. Defaults to 128.

        """
        self.embedding_model = SentenceTransformer(
            model_name_or_path=model_name, device=DEVICE
        )
        self.batch_size = batch_size

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Run inference for one batch.

        Args:
            batch: Batch containing text

        Returns:
            Dictionary that adds dense embedding to existing batch

        """
        embeddings = self.embedding_model.encode(
            sentences=batch["text"], batch_size=self.batch_size, device=DEVICE
        )
        return {
            "text": batch["text"],
            "source": batch["source"],
            "dense_vector": embeddings.tolist(),
        }

    def embed_query(self, query: str) -> list[float]:
        """
        Get embedding for query.

        Args:
            query: Query from user.

        Returns:
            List of floats as embedding vector.

        """
        return self.embedding_model.encode(
            sentences=query, batch_size=self.batch_size, device=DEVICE
        ).tolist()


class SparseEmbedChunks:
    """
    Sparse Embedding using BGE-M3 model.

    Attributes:
        sparse_embedding_model: Sparse Embedding model
            loaded using pymilvus-model library.

    """

    def __init__(self, batch_size: int = 128) -> None:
        """
        Init function for SparseEmbedChunks class.

        Args:
            batch_size: Batch size. Defaults to 128.

        """
        # Only BGM-3 sparse embedding model is supported.
        self.sparse_embedding_model = BGEM3EmbeddingFunction(
            model_name="BAAI/bge-m3",
            batch_size=batch_size,
            return_dense=False,
            device=DEVICE,
        )

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Run inference for one batch.

        Args:
            batch: Batch containing text

        Returns:
            Dictionary that adds sparse embedding to existing batch

        """
        sparse_arr = self.sparse_embedding_model.encode_documents(
            documents=batch["text"].tolist()
        )
        sparse_embeddings = [
            sparse_to_dict(sparse_array) for sparse_array in sparse_arr["sparse"]
        ]
        return {
            "text": batch["text"],
            "source": batch["source"],
            "dense_vector": batch["dense_vector"],
            "sparse_vector": sparse_embeddings,
        }

    def embed_query(self, query: str) -> dict[int, float]:
        """
        Get embedding for query.

        Args:
            query: Query from user.

        Returns:
            Dictionary containing sparse embedding vector.

        """
        sparse_arr = self.sparse_embedding_model.encode_queries(queries=[query])
        sparse_embeddings = [
            sparse_to_dict(sparse_array) for sparse_array in sparse_arr["sparse"]
        ]
        return sparse_embeddings


class FullTextEmbedChunks:
    """
    Full text search using BM-25 model.

    Attributes:
        bm25_model: BM25 Sparse Embedding model
            loaded using pymilvus-model library.

    """

    def __init__(
        self, corpus: Optional[list[str]] = None, path: str = "models/bm25.json"
    ) -> None:
        """
        Init function for FullTextEmbedChunks class.

        Args:
            corpus: List of text to be used as corpus
                to fit bm25 model. Default to None.
            path: Path to save bm25 model

        """
        self.bm25_model = None
        save_path = Path(path)

        # Train BM-25 model on corpus
        if corpus is not None:
            self.bm25_model = BM25EmbeddingFunction(corpus=corpus)
            # Save trained BM-25 model
            Path.mkdir(save_path.parent, exist_ok=True)
            self.bm25_model.save(path=save_path)
        # Load trained BM-25 model
        else:
            logger.debug(f"Loading saved BM-25 model from {save_path}")
            self.bm25_model = BM25EmbeddingFunction()
            self.bm25_model.load(path=save_path)

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Run inference for one batch.

        Args:
            batch: Batch containing text

        Returns:
            Dictionary that adds full text embedding to existing batch

        """
        full_text_arr = self.bm25_model.encode_documents(documents=batch["text"])
        full_text_embeddings = [
            sparse_to_dict(sparse_array) for sparse_array in full_text_arr
        ]
        if "sparse_vector" in batch:
            return {
                "text": batch["text"],
                "source": batch["source"],
                "dense_vector": batch["dense_vector"],
                "sparse_vector": batch["sparse_vector"],
                "full_text_vector": full_text_embeddings,
            }
        else:
            return {
                "text": batch["text"],
                "source": batch["source"],
                "dense_vector": batch["dense_vector"],
                "full_text_vector": full_text_embeddings,
            }

    def embed_query(self, query: str) -> dict[int, float]:
        """
        Get embedding for query.

        Args:
            query: Query from user.

        Returns:
            Dictionary containing sparse embedding vector.

        """
        full_text_arr = self.bm25_model.encode_queries(queries=[query])
        full_text_embeddings = [
            sparse_to_dict(sparse_array) for sparse_array in full_text_arr
        ]
        return full_text_embeddings
