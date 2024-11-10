"""Retrieval Planner to fetch context for LLM using various retrieval approaches."""

# Warning related to star imports
# ruff: noqa: F403 F405

from typing import Any

import torch
from loguru import logger
from milvus_model.reranker import BGERerankFunction

from agentic_rag.configs.config import *
from agentic_rag.data_preprocess.embed import (
    EmbedChunks,
    FullTextEmbedChunks,
    SparseEmbedChunks,
)
from agentic_rag.utils import rrf
from agentic_rag.vector_store.milvus import CustomMilvusClient

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RetrievalPlanner:
    """
    RetrievalPlanner class returns context based on retrieval approach.

    Attributes:
        collection_name: Name of collection
        top_k: Number of results to use as context
        milvus_client: Milvus vector database client
        dense_model: Dense model
        sparse_model: Sparse model
        fulltext_model: BM-25 model
        reranker_model: Re-ranker model

    """

    def __init__(self, collection_name: str, top_k: int):
        """
        Init function for RetrievalPlanner class.

        Args:
            collection_name: Name of collection
            top_k: Number of results to use as context

        """
        self.collection_name = collection_name
        self.top_k = top_k
        self.milvus_client = CustomMilvusClient()
        self.dense_model = None
        self.sparse_model = None
        self.fulltext_model = None
        self.reranker_model = None

    def _get_embedding_model(self, embedding_approach: str) -> Any:
        """
        Get embedding model based on input embedding approach.

        Args:
            embedding_approach: Embedding model to select
                Possible choices: ["dense", "sparse", "full_text"]

        Returns:
            Embedding model class.

        """
        if embedding_approach == "dense":
            return EmbedChunks(model_name=EMBEDDING_MODEL_NAME, batch_size=BATCH_SIZE)
        if ENABLE_SPARSE_INDEX and embedding_approach == "sparse":
            return SparseEmbedChunks(batch_size=BATCH_SIZE)
        if ENABLE_FULL_TEXT_INDEX and embedding_approach == "full_text":
            # Loads a trained BM-25 model
            return FullTextEmbedChunks(corpus=None)

    def no_retrieval(self):
        """No retrieval from vector store."""
        return ""

    def semantic_search_retrieval(self, query: str) -> list[str]:
        """
        Semantic search retrieval.

        Args:
            query: Input user query

        Returns:
            List of texts from vector database closest to input query.

        """
        if self.dense_model is None:
            self.dense_model = self._get_embedding_model(embedding_approach="dense")

        # Get the dense embedding vector for the query
        query_embedding = self.dense_model.embed_query(query=query)

        # Define the search parameters (distance metric used to compare two vectors)
        dense_search_params = {"metric_type": DENSE_METRIC, "params": {}}

        # Get the TOP_K closest documents to the query
        retrieved_docs = self.milvus_client.dense_search(
            collection_name=self.collection_name,
            query_dense_embedding=query_embedding,
            top_k=self.top_k,
            dense_search_params=dense_search_params,
        )
        logger.debug(f"Retrieved docs: {retrieved_docs}")
        return [res["entity"]["text"] for res in retrieved_docs]

    def sparse_search_retrieval(self, query: str) -> list[str]:
        """
        Sparse search retrieval.

        Args:
            query: Input user query

        Returns:
            List of texts from vector database closest to input query.

        """
        if self.sparse_model is None:
            self.sparse_model = self._get_embedding_model(embedding_approach="sparse")

        # Get the sparse embedding vector for the query
        query_embedding = self.sparse_model.embed_query(query=query)

        # Define the search parameters (distance metric used to compare two vectors)
        sparse_search_params = {"metric_type": SPARSE_METRIC, "params": {}}

        # Get the TOP_K closest documents to the query
        retrieved_docs = self.milvus_client.sparse_search(
            collection_name=self.collection_name,
            query_sparse_embedding=query_embedding,
            top_k=self.top_k,
            sparse_search_params=sparse_search_params,
        )
        logger.debug(f"Retrieved docs: {retrieved_docs}")
        return [res["entity"]["text"] for res in retrieved_docs]

    def fulltext_search_retrieval(self, query: str) -> list[str]:
        """
        Full text search retrieval.

        Args:
            query: Input user query

        Returns:
            List of texts from vector database closest to input query.

        """
        if self.fulltext_model is None:
            self.fulltext_model = self._get_embedding_model(
                embedding_approach="full_text"
            )

        # Get the sparse embedding vector for the query using BM-25 model
        query_embedding = self.fulltext_model.embed_query(query=query)

        # Define the search parameters (distance metric used to compare two vectors)
        sparse_search_params = {"metric_type": SPARSE_METRIC, "params": {}}

        # Get the TOP_K closest documents to the query
        retrieved_docs = self.milvus_client.full_text_search(
            collection_name=self.collection_name,
            query_full_text_embedding=query_embedding,
            top_k=self.top_k,
            sparse_search_params=sparse_search_params,
        )
        logger.debug(f"Retrieved docs: {retrieved_docs}")
        return [res["entity"]["text"] for res in retrieved_docs]

    def hybrid_search_retrieval_with_rrf(self, query: str) -> list[str]:
        """
        Hybrid search retrieval using rrf.

        Hybrid search consists of two retrievers:
            1. Dense embedding search (semantic/embedding search)
            2. Sparse embedding search (BM-25/any sparse model)

        Result collector approach (RRF) : Used to combine the results
        of both the retrieval approaches.

        Args:
            query: Input user query

        Returns:
            List of texts from vector database closest to input query.

        """
        combined_docs = []
        dense_docs = self.semantic_search_retrieval(query=query)
        combined_docs.append(dense_docs)
        if ENABLE_SPARSE_INDEX:
            sparse_docs = self.sparse_search_retrieval(query=query)
            combined_docs.append(sparse_docs)
        if ENABLE_FULL_TEXT_INDEX:
            fulltext_docs = self.fulltext_search_retrieval(query=query)
            combined_docs.append(fulltext_docs)

        rrf_docs = rrf(combined_docs)[: self.top_k]
        logger.debug(f"Retrieved docs: {rrf_docs}")
        return [res[0] for res in rrf_docs]

    def hybrid_search_retrieval_with_reranker(self, query: str) -> list[str]:
        """
        Hybrid search retrieval using reranker.

        Hybrid search consists of two retrievers:
            1. Dense embedding search (semantic/embedding search)
            2. Sparse embedding search (BM-25/any sparse model)

        Result collector approach (Re-ranker model) : Used to combine the results
        of both the retrieval approaches.

        Args:
            query: Input user query

        Returns:
            List of texts from vector database closest to input query.

        """
        combined_docs = []
        dense_docs = self.semantic_search_retrieval(query=query)
        combined_docs = dense_docs.copy()
        if ENABLE_SPARSE_INDEX:
            sparse_docs = self.sparse_search_retrieval(query=query)
            combined_docs.extend(sparse_docs)
        if ENABLE_FULL_TEXT_INDEX:
            fulltext_docs = self.fulltext_search_retrieval(query=query)
            combined_docs.extend(fulltext_docs)
        # Remove duplicates if any
        combined_docs = list(set(combined_docs))

        if self.reranker_model is None:
            self.reranker_model = BGERerankFunction(
                model_name="BAAI/bge-reranker-v2-m3",
                device=DEVICE,
                batch_size=BATCH_SIZE,
            )
        rereanker_docs = self.reranker_model(
            query=query, documents=combined_docs, top_k=self.top_k
        )
        logger.debug(f"Retrieved docs: {rereanker_docs}")
        return [res.text for res in rereanker_docs]

    def hyde_retrieval(self, hypothetical_docs: str) -> list[str]:
        """
        HyDE retrieval.

        This approach generates a fake document using LLM that answers
        the query. This fake document is used to find similar document
        in the vector store to be used as the context.

        Args:
            hypothetical_docs: Hypothetical document generated by LLM.

        Returns:
            List of texts from vector database closest to input query.

        """
        return self.semantic_search_retrieval(query=hypothetical_docs)
