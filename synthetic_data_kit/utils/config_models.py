from typing import Dict, Optional

from pydantic import BaseModel, Field

from synthetic_data_kit.utils.config import load_config


class VLLMConfig(BaseModel):
    api_base: str = "http://localhost:8000/v1"
    port: int = 8000
    model: str = "meta-llama/Llama-3.3-70B-Instruct"
    max_retries: int = 3
    retry_delay: float = 1.0


class APIEndpointConfig(BaseModel):
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    model: str = "gpt-4o"
    max_retries: int = 3
    retry_delay: float = 1.0


class LLMConfig(BaseModel):
    provider: str = "vllm"


class GenerationConfig(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.95
    chunk_size: int = 4000
    overlap: int = 200
    max_tokens: int = 4096
    num_pairs: int = 25
    num_cot_examples: int = 5
    num_cot_enhance_examples: Optional[int] = None
    batch_size: int = 32


class CurateConfig(BaseModel):
    threshold: float = 7.0
    batch_size: int = 32
    inference_batch: int = 32
    temperature: float = 0.1


class FormatConfig(BaseModel):
    default: str = "jsonl"
    include_metadata: bool = True
    pretty_json: bool = True


class AppConfig(BaseModel):
    paths: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    api_endpoint: APIEndpointConfig = Field(default_factory=APIEndpointConfig, alias="api-endpoint")
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    curate: CurateConfig = Field(default_factory=CurateConfig)
    format: FormatConfig = Field(default_factory=FormatConfig)
    prompts: Dict[str, str] = Field(default_factory=dict)

    class Config:
        allow_population_by_field_name = True


def load_config_model(path: Optional[str] = None) -> AppConfig:
    """Load YAML configuration into an ``AppConfig`` instance."""
    data = load_config(path)
    return AppConfig.model_validate(data)
