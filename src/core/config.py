import os
from dataclasses import dataclass, field
from typing import Optional
from src.core.llm_client import LLMClient

# Automatically load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: 'python-dotenv' is not installed. Using default values.")



@dataclass
class Config:
    """System-wide configuration settings and model allocations."""

    # HydraDB Connection
    hydradb_bolt_uri: str = field(
        default_factory=lambda: os.getenv("HYDRADB_BOLT_URI", "bolt://127.0.0.1:7687")
    )
    hydradb_auth_token: str = field(
        default_factory=lambda: os.getenv("HYDRADB_AUTH_TOKEN", "")
    )

    # Extractor Model Config
    extractor_base_url: str = field(
        default_factory=lambda: os.getenv("EXTRACTOR_BASE_URL", "https://api.fireworks.ai/inference/v1")
    )
    extractor_api_key: str = field(
        default_factory=lambda: os.getenv("EXTRACTOR_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    )
    extractor_model: str = field(
        default_factory=lambda: os.getenv("EXTRACTOR_MODEL", "gpt-oss-20b")
    )

    # Reader Model Config
    reader_base_url: str = field(
        default_factory=lambda: os.getenv("READER_BASE_URL", "https://api.fireworks.ai/inference/v1")
    )
    reader_api_key: str = field(
        default_factory=lambda: os.getenv("READER_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    )
    reader_model: str = field(
        default_factory=lambda: os.getenv("READER_MODEL", "gpt-oss-120b")
    )

    # Judge Model Config
    judge_base_url: str = field(
        default_factory=lambda: os.getenv("JUDGE_BASE_URL", "https://openrouter.ai/api/v1")
    )
    judge_api_key: str = field(
        default_factory=lambda: os.getenv("JUDGE_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
    )
    judge_model: str = field(
        default_factory=lambda: os.getenv("JUDGE_MODEL", "meta-llama/llama-3.3-70b-instruct")
    )

    # Retrieval & Engine Constants (per AGENT.md)
    temporal_buffer_seconds: float = 172800.0  # ±2 days
    semantic_top_k: int = 10
    semantic_weight: float = 0.6
    structural_weight: float = 0.4
    reader_top_n: int = 15
    abstention_cutoff: float = 0.25
    abstention_response: str = "I don't have that information in my memory."

    def get_extractor_client(self) -> LLMClient:
        """Construct an LLMClient configured for extraction tasks."""
        return LLMClient(
            base_url=self.extractor_base_url,
            api_key=self.extractor_api_key,
            model_name=self.extractor_model,
        )

    def get_reader_client(self) -> LLMClient:
        """Construct an LLMClient configured for reader/QA tasks."""
        return LLMClient(
            base_url=self.reader_base_url,
            api_key=self.reader_api_key,
            model_name=self.reader_model,
        )

    def get_judge_client(self) -> LLMClient:
        """Construct an LLMClient configured for judge evaluation tasks."""
        return LLMClient(
            base_url=self.judge_base_url,
            api_key=self.judge_api_key,
            model_name=self.judge_model,
        )


# Global default configuration instance
config = Config()
