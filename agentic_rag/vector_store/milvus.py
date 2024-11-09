"""Milvus vector database."""

from typing import Any

import numpy as np
from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)


def create_schema(
    dense_dim: int,
    add_sparse_index: bool = False,
    add_full_text_index: bool = False,
) -> CollectionSchema:
    """Create a schema for Milvus vector store.

    Args:
        dense_dim: Embedding model output dimension.
        add_sparse_index: Enable sparse indexing.
                Defaults to False.
        add_full_text_index: Enable full text indexing.
                Defaults to False.

    Returns:
        CollectionSchema for Milvus vector store.

    """
    # Specify the data schema for the new Collection.
    fields = [
        # Use auto generated id as primary key
        FieldSchema(
            name="pk",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=True,
            max_length=100,
        ),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=65535),
    ]
    if add_sparse_index:
        fields.append(
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)
        )
    if add_full_text_index:
        fields.append(
            FieldSchema(name="full_text_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)
        )
    schema = CollectionSchema(fields, "")
    return schema


class CustomMilvusClient:
    """Custom Milvus client."""

    def __init__(self) -> None:
        """Connect to milvus lite client."""
        self.milvus_client = MilvusClient(uri="./milvus.db")

    def create_collection(
        self,
        collection_name: str,
        dense_dim: int,
        dense_distance_metric: str,
        add_sparse_index: bool = False,
        add_full_text_index: bool = False,
    ) -> None:
        """Create a collection for a milvus vector store.

        Args:
            collection_name: Name of the collection
            dense_dim: Embedding dimension for dense vector
            dense_distance_metric: Metric used by dense embedding model
            add_sparse_index: Enable sparse indexing. Defaults to False.
            add_full_text_index: Enable full text indexing.
                    Defaults to False.

        """
        # Drop if collection exists
        has_collection = self.milvus_client.has_collection(collection_name, timeout=5)
        if has_collection:
            self.milvus_client.drop_collection(collection_name)

        index_params = self.milvus_client.prepare_index_params()
        if add_sparse_index:
            # scalar index
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_inverted_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
            )
        if add_full_text_index:
            # scalar index
            index_params.add_index(
                field_name="full_text_vector",
                index_name="sparse_inverted_index_full_text",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
            )
        # vector index
        index_params.add_index(
            field_name="dense_vector",
            index_name="flat",
            index_type="FLAT",
            metric_type=dense_distance_metric,
        )
        schema = create_schema(dense_dim, add_sparse_index, add_full_text_index)

        self.milvus_client.create_collection(
            collection_name,
            schema=schema,
            index_params=index_params,
        )

    def store(
        self,
        data: list[dict[str, Any]],
        collection_name: str,
    ) -> None:
        """Store texts, embedding (dense and sparse) and source.

        Args:
            data: Dataset to be stored in milvus vector store.
            collection_name: Name of the collection

        """
        _ = self.milvus_client.insert(collection_name, data, progress_bar=True)

    def dense_search(
        self,
        collection_name: str,
        query_dense_embedding: list[float],
        top_k: int,
        dense_search_params: dict,
    ) -> list[dict]:
        """Perform dense search.

        Args:
            collection_name: Name of the collection
            query_dense_embedding: The embedding vector from the
                dense model for the query as a list of floats.
            top_k: Top k entries semantically similar to query
                in the vector store.
            dense_search_params: The search parameters such as
                metrics and other param used to calculate score.

        Returns:
            List of dictionary containing text, index and score sorted by score.

        """
        result = self.milvus_client.search(
            collection_name=collection_name,
            data=[query_dense_embedding],
            anns_field="dense_vector",
            limit=top_k,
            output_fields=["text"],
            search_params=dense_search_params,
        )
        return result[0]

    def sparse_search(
        self,
        collection_name: str,
        query_sparse_embedding: list[dict[int, float]],
        top_k: int,
        sparse_search_params: dict,
    ) -> list[dict]:
        """Perform sparse search.

        Args:
            collection_name: Name of the collection
            query_sparse_embedding: The sparse vector from the
                sparse model for the query
            top_k: Top k entries semantically similar to query
                in the vector store.
            sparse_search_params: The search parameters such as
                metrics and other param used to calculate score.

        Returns:
            List of dictionary containing text, index and score sorted by score.

        """
        result = self.milvus_client.search(
            collection_name=collection_name,
            data=query_sparse_embedding,
            anns_field="sparse_vector",
            limit=top_k,
            output_fields=["text"],
            search_params=sparse_search_params,
        )
        return result[0]

    def full_text_search(
        self,
        collection_name: str,
        query_full_text_embedding: list[dict[int, float]],
        top_k: int,
        sparse_search_params: dict,
    ) -> list[dict]:
        """Perform full text search using BM25 model.

        Args:
            collection_name: Name of the collection
            query_full_text_embedding: The full text vector from the
                BM25 model for the query
            top_k: Top k entries semantically similar to query
                in the vector store.
            sparse_search_params: The search parameters such as
                metrics and other param used to calculate score.

        Returns:
            List of dictionary containing text, index and score sorted by score.

        """
        result = self.milvus_client.search(
            collection_name=collection_name,
            data=query_full_text_embedding,
            anns_field="full_text_vector",
            limit=top_k,
            output_fields=["text"],
            search_params=sparse_search_params,
        )
        return result[0]
