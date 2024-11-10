"""Retrieval Engine."""

import os
from collections import defaultdict
from string import Template

from dotenv import load_dotenv
from loguru import logger

from agentic_rag.configs.config import (
    DEBUG_WITHOUT_LLM,
    LLM_API_BASE,
    LLM_MODEL,
    MAX_OUTPUT_TOKENS,
    RETRIEVAL_PRIORITY,
    STUB_RESPONSE,
    TRUST_SCORE_THRESH,
)
from agentic_rag.configs.prompts import HYDE_PROMPT, LLM_SYSTEM_PROMPT, RAG_PROMPT
from agentic_rag.generation.llm_engine import LLMEngine
from agentic_rag.retrieval.retrieval_agent_planner import RetrievalPlanner
from agentic_rag.utils import timeit

# Load API keys from .env file as environment variable
load_dotenv()


class RetrievalEngine:
    """
    Retrieval Engine.

    Retrieval Engine is responsible for creating a retrieval plan
    and providing a response from LLM based on uncertaininty score.

    Attributes:
        llm_engine: LLM engine to get response from LLM
        query: Input query
        planner: Retrieval planner that implements several retrieval
            strategies

    """

    def __init__(self, query: str, collection_name: str, top_k: int) -> None:
        """
        Init function for RetrievalEngine class.

        Args:
            query: Input query.
            collection_name: Name of vector db collection
            top_k: Number of results to be used as context.

        """
        self.llm_engine = None
        self.query = query
        self.planner = RetrievalPlanner(collection_name=collection_name, top_k=top_k)

    def _get_retriever(
        self,
        retrieval_strategy: str,
        hyde_prompt: str = HYDE_PROMPT,
        max_tokens: int = MAX_OUTPUT_TOKENS,
        system_prompt: str = LLM_SYSTEM_PROMPT,
    ):
        """
        Get documents based on retrieval strategy.

        Args:
            retrieval_strategy: Selected retrieval strategy.
            hyde_prompt: Prompt used by HyDE retrieval approach
            max_tokens: Maximum output token from LLM response
            system_prompt: System prompt passed to LLM alongside user prompt

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
        if retrieval_strategy == "hyde_retrieval":
            if DEBUG_WITHOUT_LLM:
                logger.warning(
                    "HyDE retrieval requires LLM to fetch similar documents."
                )
                return ""
            _, response_docs = self.get_llm_response(
                context=self.query,
                prompt_template=hyde_prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
            return self.planner.hyde_retrieval(hypothetical_docs=response_docs)

    def get_context(self, retrieval_strategy: str) -> str:
        """
        Get the context for LLM based on retrieval strategy.

        Args:
            retrieval_strategy: Selected retrieval strategy.

        Returns:
            The retrieved context as a single concaatenated string
            using selected retrieval strategy.

        """
        # TODO: pass system_message and max_tokens to _get_retriever
        logger.info(f"Using retrieval strategy = {retrieval_strategy}")
        retrieved_texts = self._get_retriever(retrieval_strategy)
        return " ".join(retrieved_texts)

    def _init_llm_engine(self) -> None:
        """Initialize LLM engine."""
        if self.llm_engine is None:
            self.llm_engine = LLMEngine(llm_model=LLM_MODEL, llm_api_base=LLM_API_BASE)

    def _get_llm_prompt(self, prompt_template: str, context: str) -> str:
        """
        Create a LLM prompt based on inputs required by prompt_template.

        Args:
            prompt_template: Input prompt template
            context: Context to be added to the prompt template.

        Returns:
            LLM prompt.

        """
        llm_prompt = Template(prompt_template)
        if set(llm_prompt.get_identifiers()) == set(["context", "query"]):
            return llm_prompt.substitute(query=self.query, context=context)
        else:
            return llm_prompt.substitute(query=self.query)

    def pretty_table(self, results_dict: dict) -> None:
        """
        Create a Table using `rich` library to summarize the outputs.

        Table prints three pieces of information
        1. Retrieval strategy
        2. Trustworthyness score
        3. LLM Response

        Args:
            results_dict: Dictionary containing information about
                retrieval strategy, corresponding trust score and llm response.

        """
        from rich.console import Console
        from rich.table import Table

        table = Table(title=f"Summary for query: {self.query}", show_lines=True)
        table.add_column(
            "Retrieval Strategy", justify="left", style="cyan", no_wrap=True
        )
        table.add_column("Trustworthy score", style="magenta")
        table.add_column("LLM Response", style="green")
        for key, value in results_dict.items():
            table.add_row(key, str(value[0]), value[1])
        console = Console()
        console.print(table)

    def get_llm_response(
        self, context: str, prompt_template: str, max_tokens: int, system_prompt: str
    ) -> tuple[str, str]:
        """
        Get LLM response for given prompt.

        Construct LLM prompt using prompt_template for given context
        and input query.

        Args:
            context: Context to be sent to LLM.
            prompt_template: Prompt template to be used for the retrieved context
                and user question.
            max_tokens: Maximum output token from LLM response
            system_prompt: System prompt passed to LLM alongside user prompt

        Returns:
            A tuple of strings containing llm prompt and response.

        """
        llm_prompt = self._get_llm_prompt(
            prompt_template=prompt_template, context=context
        )
        logger.debug(f"LLM Prompt:\n{llm_prompt}")
        if DEBUG_WITHOUT_LLM:
            return llm_prompt, ""
        self._init_llm_engine()
        llm_response = self.llm_engine.generate_response(
            user_prompt=llm_prompt, max_tokens=max_tokens, system_prompt=system_prompt
        )
        logger.info(f"Query: {self.query}")
        logger.info(f"LLM Response:\n{llm_response}")
        return llm_prompt, llm_response

    def calculate_uncertainity(self, prompt: str, response: str) -> float:
        """
        Provide a uncertainity score for LLM response and prompt.

        There are various approaches to get a score such as
        Trustworthy Language Model, BSDetector, SelfCheckGPT, or Prometheus 2.

        Here we use Trustworthy Language Model that requires a API Key.

        Args:
            prompt: Prompt provided to LLM to get a response
            response: Response from LLM

        Returns:
            Uncertainity score.
            If DEBUG_WITHOUT_LLM is set, return 0 score.

        """
        if DEBUG_WITHOUT_LLM:
            return 0

        from cleanlab_studio import Studio

        if os.environ.get("CLEANLAB_API_KEY", None) is None:
            raise ValueError(
                "Get API key from here: "
                "https://app.cleanlab.ai/account after creating an account."
            )
        else:
            studio = Studio(api_key=os.environ["CLEANLAB_API_KEY"])
            tlm = studio.TLM()
            response = tlm.get_trustworthiness_score(prompt, response=response)
            trustworthiness_score = response["trustworthiness_score"]
            logger.debug(f"Trustworthiness score = {trustworthiness_score}")
            return trustworthiness_score

    @timeit
    def user_plan(
        self,
        prompt_template: str = RAG_PROMPT,
        max_tokens: int = MAX_OUTPUT_TOKENS,
        system_prompt: str = LLM_SYSTEM_PROMPT,
    ) -> str:
        """
        Provide a LLM response based on uncertaininty score and retrieval strategies.

        A list of prioritised retrieval strategies based on cost and complexity is provided.
        For a particular selected retrieval strategy, we estimate the trustworthiness score,
        if `TRUST_SCORE_THRESH` is not met, next retrieval strategy is used to get the response.

        If no Retrieval strategy yields a trustworthy LLM response,
        then our system responds with a `STUB_RESPONSE`.

        Args:
            prompt_template: Prompt template to be used for the retrieved context
                and user question.
            max_tokens: Maximum output token from LLM response
            system_prompt: System prompt passed to LLM alongside user prompt

        Returns:
            LLM Response or STUB Response

        """  # noqa: E501
        results_dict = defaultdict()
        for retrieval_strategy in RETRIEVAL_PRIORITY:
            context = self.get_context(retrieval_strategy)
            llm_prompt, response = self.get_llm_response(
                context=context,
                prompt_template=prompt_template,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
            trustworthiness_score = self.calculate_uncertainity(
                prompt=llm_prompt, response=response
            )
            results_dict[retrieval_strategy] = (trustworthiness_score, response)

            if trustworthiness_score >= TRUST_SCORE_THRESH:
                self.pretty_table(results_dict)
                return response

        self.pretty_table(results_dict)
        return STUB_RESPONSE

    def agent_plan(self):
        """
        Instead of user planning the retrieval strategies, an agent decides.

        Raises:
            NotImplementedError: Add a agentic planner

        """
        raise NotImplementedError("Agentic planning not implemented.")
