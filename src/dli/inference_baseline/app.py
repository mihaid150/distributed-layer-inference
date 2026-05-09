from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from dli.common.logging_config import configure_logging
from dli.common.schemas import GenerateRequest, GenerateResponse, HealthResponse
from dli.inference_baseline.baseline_config import BaselineConfig, load_baseline_config
from dli.inference_baseline.baseline_executor import BaselineExecutor
from dli.inference_baseline.model_loader import BaselineModelLoader


baseline_config: BaselineConfig = load_baseline_config()
logger = configure_logging(baseline_config.service_name)

logger.info("Loading baseline model: %s", baseline_config.model_name)

loader = BaselineModelLoader(
    model_name=baseline_config.model_name,
    device=baseline_config.device,
    torch_dtype=baseline_config.torch_dtype,
)

tokenizer, model = loader.load()

executor = BaselineExecutor(
    tokenizer=tokenizer,
    model=model,
    service_name=baseline_config.service_name,
    device=baseline_config.device,
)

app = FastAPI(
    title="Inference Baseline",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service_name=baseline_config.service_name,
        stage_id=None,
    )


@app.get("/config")
def config() -> Dict[str, Any]:
    return {
        "service_name": baseline_config.service_name,
        "model_name": baseline_config.model_name,
        "device": baseline_config.device,
        "torch_dtype": baseline_config.torch_dtype,
        "execution_mode": "single_node_baseline",
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    prompt_text = request.prompt or ""
    if not prompt_text and request.messages:
        for message in reversed(request.messages):
            if (message.role or "").strip().lower() == "user":
                prompt_text = message.content or ""
                break
        if not prompt_text:
            prompt_text = request.messages[-1].content or ""

    logger.info(
        "Received baseline generation request max_new_tokens=%s prompt_length_chars=%s",
        request.max_new_tokens,
        len(prompt_text),
    )

    try:
        return executor.generate(request)

    except RuntimeError as exc:
        logger.exception("Baseline generation failed due to runtime error.")
        raise HTTPException(
            status_code=500,
            detail=f"Baseline generation failed: {exc}",
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected baseline generation failure.")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected baseline generation failure: {exc}",
        ) from exc
