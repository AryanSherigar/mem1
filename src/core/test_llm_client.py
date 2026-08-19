import pytest
from src.core.llm_client import LLMClient

#write a test for a function/class that builds an OpenAI-compatible client pointed at Fireworks' base URL, using an API key.
def test_build_client_uses_fireworks_base_url():
    client = LLMClient(base_url="https://api.fireworks.ai/inference/v1", api_key="fake-key", model_name="gpt-oss-20b")
 
    assert str(client.base_url) == "https://api.fireworks.ai/inference/v1/"


def test_build_client_uses_given_api_key():
    client = LLMClient(base_url="https://api.fireworks.ai/inference/v1", api_key="fake-key", model_name="gpt-oss-20b")

    assert client.api_key == "fake-key"
