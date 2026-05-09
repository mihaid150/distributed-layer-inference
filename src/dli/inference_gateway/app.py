from __future__ import annotations

import os
from typing import Any, Dict

import yaml
from fastapi import FastAPI, HTTPException
from requests import RequestException
from transformers import AutoTokenizer

from dli.common.logging_config import configure_logging
from dli.common.schemas import ChatRequest, GenerateRequest, GenerateResponse, HealthResponse
from dli.inference_gateway.generation_loop import GenerationLoop
from dli.inference_gateway.request_schema import GatewayConfig
from dli.inference_gateway.stage_client import StageClient


def load_gateway_config() -> GatewayConfig:
    config_path = os.getenv("STAGE_MAP_PATH", "/app/configs/stage_map.yaml")

    config_data: Dict[str, Any] = {}

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file) or {}

    gateway_data = config_data.get("inference_gateway", {})

    first_stage_url = os.getenv(
        "FIRST_STAGE_URL",
        gateway_data.get("first_stage_url", "http://inference-stage-1:8000/forward"),
    )

    model_name = os.getenv(
        "MODEL_NAME",
        config_data.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    )

    return GatewayConfig(
        service_name=gateway_data.get("service_name", "inference-gateway"),
        first_stage_url=first_stage_url,
        model_name=model_name,
    )


gateway_config = load_gateway_config()
logger = configure_logging(gateway_config.service_name)

logger.info("Loading tokenizer: %s", gateway_config.model_name)
tokenizer = AutoTokenizer.from_pretrained(gateway_config.model_name)

stage_client = StageClient(first_stage_url=gateway_config.first_stage_url)

generation_loop = GenerationLoop(
    tokenizer=tokenizer,
    stage_client=stage_client,
)

app = FastAPI(
    title="Inference Gateway",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service_name=gateway_config.service_name,
        stage_id=None,
    )


@app.get("/config")
def config() -> Dict[str, Any]:
    return {
        "service_name": gateway_config.service_name,
        "model_name": gateway_config.model_name,
        "first_stage_url": gateway_config.first_stage_url,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    prompt_text = (request.prompt or "").strip()
    if not prompt_text and request.messages:
        for message in reversed(request.messages):
            if (message.role or "").strip().lower() == "user":
                prompt_text = (message.content or "").strip()
                break
        if not prompt_text:
            prompt_text = (request.messages[-1].content or "").strip()

    logger.info(
        "Received generation request max_new_tokens=%s prompt_length_chars=%s",
        request.max_new_tokens,
        len(prompt_text),
    )

    try:
        return generation_loop.generate(request)

    except RequestException as exc:
        logger.exception("Failed to communicate with Inference Stage pipeline.")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to communicate with Inference Stage pipeline: {exc}",
        ) from exc

    except Exception as exc:
        logger.exception("Generation failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {exc}",
        ) from exc


@app.post("/chat", response_model=GenerateResponse)
def chat(request: ChatRequest) -> GenerateResponse:
    logger.info(
        "Received chat request turns=%s max_new_tokens=%s",
        len(request.messages),
        request.max_new_tokens,
    )

    generation_request = GenerateRequest(
        prompt=None,
        messages=request.messages,
        max_new_tokens=request.max_new_tokens,
        min_new_tokens=request.min_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
    )

    try:
        return generation_loop.generate(generation_request)

    except RequestException as exc:
        logger.exception("Failed to communicate with Inference Stage pipeline.")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to communicate with Inference Stage pipeline: {exc}",
        ) from exc

    except Exception as exc:
        logger.exception("Chat generation failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Chat generation failed: {exc}",
        ) from exc
