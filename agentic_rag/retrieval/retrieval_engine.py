from agentic_rag.configs.config import *
from agentic_rag.configs.prompts import RAG_PROMPT
from agentic_rag.retrieval.retrieval_agent_planner import RetrievalPlanner
from loguru import logger


class RetrievalEngine:

    def __init__(self, query: str, collection_name: str, top_k: int) -> None:
        self.llm_engine = None
        self.query = query
        self.planner = RetrievalPlanner(collection_name=collection_name, top_k=top_k)

    def _get_retriever(self, retrieval_strategy: str):
        """Get documents based on retrieval strategy.

        Args:
            retrieval_strategy: Selected retrieval strategy.

        Returns:
            Documents retrieved based on selected retrieval strategy
        """
        if retrieval_strategy == "no_retrieval":
            return self.planner.no_retrieval()
        if retrieval_strategy == "semantic_search":
            return self.planner.semantic_search_retrieval(query=self.query)
        if retrieval_strategy == "hybrid_search_with_rrf":
            return self.planner.hybrid_search_retrieval_with_rrf(query=self.query)
        if retrieval_strategy == "hybrid_search_with_reranker":
            return self.planner.hybrid_search_retrieval_with_reranker(query=self.query)

    def get_context(self, retrieval_strategy: str) -> str:
        """Get the context for LLM based on retrieval strategy

        Args:
            retrieval_strategy: Selected retrieval strategy.

        Returns:
            _description_
        """
        logger.info(f"Using retrieval strategy = {retrieval_strategy}")
        retrieved_docs = self._get_retriever(retrieval_strategy)
        logger.debug(f"Retrieved docs: {retrieved_docs}")

    def get_llm_response(self, context: str, prompt_template: str) -> tuple[str, str]:
        """_summary_

        Args:
            context: _description_
            prompt_template: _description_

        Returns:
            _description_
        """
        llm_prompt = prompt_template.format(query=self.query, context=context)
        if self.llm_engine is None:
            pass
        response = ""
        return llm_prompt, response

    def calculate_uncertainty(self, prompt: str, response: str) -> float:
        return 0

    def user_plan(self) -> str:
        """Provide a LLM response based on uncertaininty score and retrieval strategies.

        We provide a list of prioritised retrieval strategies based on cost and complexity.
        For a particular selected retrieval strategy, we estimate the trustworthiness score,
        if `TRUST_SCORE_THRESH` is not met, next retrieval strategy is used to get the response.

        If no Retrieval strategy yields a trustworthy LLM response,
        then our system responds with a `STUB_RESPONSE`.

        Returns:
            LLM Response or STUB Response
        """
        for retrieval_strategy in RETRIEVAL_PRIORITY:
            context = self.get_context(retrieval_strategy)
            llm_prompt, response = self.get_llm_response(
                context=context, prompt_template=RAG_PROMPT
            )
            uncertainity_score = self.calculate_uncertainty(
                prompt=llm_prompt, response=response
            )
            if uncertainity_score >= TRUST_SCORE_THRESH:
                return response
        return STUB_RESPONSE

    def agent_plan(self):
        """Instead of user planning the retrieval strategies, an agent decides.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("Agentic planning not implemented.")
