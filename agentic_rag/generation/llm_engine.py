"""LLM engine."""

from typing import Optional
import litellm
from dotenv import load_dotenv
from litellm.utils import (
    validate_environment,
    get_response_string,
    supports_system_messages,
)
from loguru import logger

# Load API keys from .env file as environment variable
load_dotenv()


class LLMEngine:
    """LLM Engine.

    Attributes:
        llm_model:
        llm_api_base:
    """

    def __init__(self, llm_model: str, llm_api_base: Optional[str] = None) -> None:
        """_summary_

        Args:
            llm_model: _description_
            llm_api_base: _description_. Defaults to None.

        """
        self.llm_model = llm_model
        self.llm_api_base = llm_api_base
        self._validate_api_key(llm_model)

    def _validate_api_key(self, llm_model: str) -> None:
        """Validate if correct API key are loaded for the model.

        Args:
            llm_model: _description_

        Raises:
            ValueError: _description_
        """
        response = validate_environment(model=llm_model)
        if not response["keys_in_environment"]:
            raise ValueError(
                f"Following API key required by model are not set: {response['missing_keys']}"
            )

    def generate_response(
        self, user_prompt: str, max_tokens: int, system_prompt: str
    ) -> str:
        """_summary_

        Args:
            user_prompt: _description_
            max_tokens: _description_
            system_prompt: _description_

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
