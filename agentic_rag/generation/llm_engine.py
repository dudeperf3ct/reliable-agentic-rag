"""LLM engine."""

from typing import Optional

import litellm
from dotenv import load_dotenv
from litellm.utils import (
    get_response_string,
    supports_system_messages,
    validate_environment,
)
from loguru import logger

# Load API keys from .env file as environment variable
load_dotenv()


class LLMEngine:
    """
    LLM Engine.

    Attributes:
        llm_model: LLM model
        llm_api_base: LLM base url

    """

    def __init__(self, llm_model: str, llm_api_base: Optional[str] = None) -> None:
        """
        Init function for LLMEngine class.

        Args:
            llm_model: LLM
            llm_api_base: LLM base url. Defaults to None.

        """
        self.llm_model = llm_model
        self.llm_api_base = llm_api_base
        self._validate_api_key(llm_model)

    def _validate_api_key(self) -> None:
        """
        Validate if correct API key are loaded for the model.

        Raises:
            ValueError: If API key is not set for selected LLM

        """
        response = validate_environment(model=self.llm_model)
        if not response["keys_in_environment"]:
            raise ValueError(
                "Following API key required by model are not set: "
                f"{response['missing_keys']}"
            )

    def generate_response(
        self, user_prompt: str, max_tokens: int, system_prompt: str
    ) -> str:
        """
        Generate LLM response for given LLM prompt.

        Args:
            user_prompt: Input LLM prompt containing context
                and user query
            max_tokens: Number of output tokens
            system_prompt: System prompt to be added to LLM

        Returns:
            LLM Response.

        """
        messages = [{"role": "user", "content": f"{user_prompt}"}]
        if supports_system_messages(model=self.llm_model, custom_llm_provider=None):
            messages.append({"role": "system", "content": f"{system_prompt}"})
        response = litellm.completion(
            model=self.llm_model,
            messages=messages,
            max_tokens=max_tokens,
            api_base=self.llm_api_base,
        )
        logger.debug(f"LLM Response: {response}")
        return get_response_string(response)
