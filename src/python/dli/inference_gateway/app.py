from __future__ import annotations

import os
from typing import Any, Dict

import yaml
from fastapi import FastAPI, HTTPException
from requests import RequestException
from transformers import AutoTokenizer

from dli.common.feature_flags import FeatureFlags
from dli.common.logging_config import configure_logging
from dli.feature_modules.catalog import FEATURE_MODULE_CATALOG
from dli.feature_modules.persistent_backpressure import GatewayBackpressureGuard
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
        topology_route_candidates=_load_topology_route_candidates(config_data),
        topology_probe_timeout_seconds=float(
            (config_data.get("topology") or {}).get("probe_timeout_seconds", 0.25)
        ),
    )


def _load_topology_route_candidates(config_data: Dict[str, Any]) -> Dict[str, list[str]]:
    route_candidates: Dict[str, list[str]] = {}

    for stage in config_data.get("inference_stages", []) or []:
        stage_id = str(stage.get("stage_id", ""))
        if not stage_id:
            continue

        candidates = stage.get("route_candidates")
        if candidates is None:
            candidates = [stage.get("next_stage_url")] if stage.get("next_stage_url") else []
        if isinstance(candidates, str):
            candidates = [candidates]
        if not isinstance(candidates, list):
            continue

        route_candidates[stage_id] = [
            str(url).strip()
            for url in candidates
            if isinstance(url, str) and str(url).strip()
        ]

    topology_data = config_data.get("topology") or {}
    explicit_candidates = topology_data.get("route_candidates") or {}
    if isinstance(explicit_candidates, dict):
        for stage_id, candidates in explicit_candidates.items():
            if isinstance(candidates, str):
                candidates = [candidates]
            if not isinstance(candidates, list):
                continue
            route_candidates[str(stage_id)] = [
                str(url).strip()
                for url in candidates
                if isinstance(url, str) and str(url).strip()
            ]

    return route_candidates


gateway_config = load_gateway_config()
logger = configure_logging(gateway_config.service_name)

logger.info("Loading tokenizer: %s", gateway_config.model_name)
tokenizer = AutoTokenizer.from_pretrained(gateway_config.model_name)

stage_client = StageClient(first_stage_url=gateway_config.first_stage_url)

generation_loop = GenerationLoop(
    tokenizer=tokenizer,
    stage_client=stage_client,
    topology_route_candidates=gateway_config.topology_route_candidates,
    topology_probe_timeout_seconds=gateway_config.topology_probe_timeout_seconds,
)
backpressure_guard = GatewayBackpressureGuard()

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
        "topology_route_candidates": gateway_config.topology_route_candidates,
        "optimized_module_profile": FeatureFlags.optimized_module_profile(),
        "feature_modules": FEATURE_MODULE_CATALOG,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    queue_acquired = False
    if request.feature_flags.backpressure_enabled:
        queue_limit = int(request.feature_flags.backpressure_queue_size)
        queue_acquired = backpressure_guard.acquire(queue_limit=queue_limit)
        if not queue_acquired:
            snapshot = backpressure_guard.snapshot(queue_limit=queue_limit)
            raise HTTPException(
                status_code=429,
                detail=(
                    "Backpressure queue is full. "
                    f"inflight={snapshot.inflight} waiting={snapshot.waiting} "
                    f"queue_limit={snapshot.queue_limit}"
                ),
            )

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
    finally:
        if queue_acquired:
            backpressure_guard.release()


@app.post("/chat", response_model=GenerateResponse)
def chat(request: ChatRequest) -> GenerateResponse:
    queue_acquired = False
    if request.feature_flags.backpressure_enabled:
        queue_limit = int(request.feature_flags.backpressure_queue_size)
        queue_acquired = backpressure_guard.acquire(queue_limit=queue_limit)
        if not queue_acquired:
            snapshot = backpressure_guard.snapshot(queue_limit=queue_limit)
            raise HTTPException(
                status_code=429,
                detail=(
                    "Backpressure queue is full. "
                    f"inflight={snapshot.inflight} waiting={snapshot.waiting} "
                    f"queue_limit={snapshot.queue_limit}"
                ),
            )

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
        feature_flags=request.feature_flags,
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
    finally:
        if queue_acquired:
            backpressure_guard.release()
