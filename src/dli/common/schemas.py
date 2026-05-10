from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from dli.common.feature_flags import FeatureFlags

class ChatMessage(BaseModel):
    role: str = Field(..., description="Chat role, e.g. system/user/assistant.")
    content: str = Field(..., description="Message text.")


class GenerateRequest(BaseModel):
    prompt: Optional[str] = Field(
        default=None,
        description="Input prompt received by the Inference Gateway.",
    )
    messages: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional chat message list for conversation generation.",
    )
    max_new_tokens: int = Field(default=32, ge=1, le=512)
    min_new_tokens: int = Field(default=8, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)

    @model_validator(mode="after")
    def validate_prompt_or_messages(self) -> "GenerateRequest":
        has_prompt = bool((self.prompt or "").strip())
        has_messages = bool(self.messages)
        if not has_prompt and not has_messages:
            raise ValueError("Either prompt or messages must be provided.")
        return self


class TokenStepMetric(BaseModel):
    token_index: int
    token_id: int
    token_text: str
    latency_ms: float
    stage_metrics: List[Dict[str, Any]] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    prompt: str
    assistant_message: Optional[str] = None
    generated_text: str
    generated_token_ids: List[int]
    total_latency_ms: float
    tokens_per_second: float
    termination_reason: str = "max_new_tokens"
    prompt_token_count: int = 0
    summary_metrics: Dict[str, Any] = Field(default_factory=dict)
    token_metrics: List[TokenStepMetric]


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    min_new_tokens: int = Field(default=8, ge=1, le=512)
    temperature: float = Field(default=0.2, ge=0.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)


class StageForwardRequest(BaseModel):
    request_id: str
    token_index: int
    tensor_b64: str
    tensor_dtype: str
    tensor_shape: List[int]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    metrics: List[Dict[str, Any]] = Field(default_factory=list)
    transport: Dict[str, Any] = Field(default_factory=dict)
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)


class StageForwardResponse(BaseModel):
    request_id: str
    token_index: int
    server_wall_ms: float = 0.0
    tensor_b64: Optional[str] = None
    tensor_dtype: Optional[str] = None
    tensor_shape: Optional[List[int]] = None

    next_token_id: Optional[int] = None
    next_token_text: Optional[str] = None

    metrics: List[Dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    service_name: str
    stage_id: Optional[int] = None
