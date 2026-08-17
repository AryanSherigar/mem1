import os
from src.core.config import Config, config
from src.core.llm_client import LLMClient


def test_default_config():
    assert config.extractor_model == "gpt-oss-20b"
    assert config.reader_model == "gpt-oss-120b"
    assert config.judge_model == "meta-llama/llama-3.3-70b-instruct"
    assert config.temporal_buffer_seconds == 172800.0
    assert config.semantic_top_k == 10
    assert config.abstention_cutoff == 0.25
    assert config.abstention_response == "I don't have that information in my memory."


def test_llm_client_factories():
    cfg = Config(
        extractor_base_url="https://api.fireworks.ai/inference/v1",
        extractor_api_key="test-ext-key",
        extractor_model="gpt-oss-20b",
        reader_base_url="https://api.fireworks.ai/inference/v1",
        reader_api_key="test-reader-key",
        reader_model="gpt-oss-120b",
        judge_base_url="https://openrouter.ai/api/v1",
        judge_api_key="test-judge-key",
        judge_model="meta-llama/llama-3.3-70b-instruct",
    )

    extractor = cfg.get_extractor_client()
    assert isinstance(extractor, LLMClient)
    assert extractor.model == "gpt-oss-20b"

    reader = cfg.get_reader_client()
    assert isinstance(reader, LLMClient)
    assert reader.model == "gpt-oss-120b"

    judge = cfg.get_judge_client()
    assert isinstance(judge, LLMClient)
    assert judge.model == "meta-llama/llama-3.3-70b-instruct"
