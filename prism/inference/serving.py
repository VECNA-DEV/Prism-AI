"""FastAPI serving endpoint for Prism AI.

Provides a REST API for text generation with the trained model.
Supports streaming and non-streaming responses.

Usage:
    uvicorn prism.inference.serving:app --host 0.0.0.0 --port 8000
"""

import os
import time
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from prism.model.config import PrismConfig
from prism.model.model import PrismForCausalLM
from prism.data.tokenizer import PrismTokenizer
from prism.inference.generator import PrismGenerator


# ── API Models ──────────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    """Request body for text generation."""
    prompt: str = Field(..., description="Input prompt text")
    max_new_tokens: int = Field(512, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=0, le=1000, description="Top-k filtering (0 = disabled)")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p (nucleus) filtering")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")


class GenerateResponse(BaseModel):
    """Response body for text generation."""
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    total_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_params_billions: float
    device: str


# ── Application ─────────────────────────────────────────────────────


app = FastAPI(
    title="Prism AI",
    description="10B Parameter Foundation Model for Code Generation",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state — populated by load_model()
_generator: Optional[PrismGenerator] = None
_tokenizer: Optional[PrismTokenizer] = None
_config: Optional[PrismConfig] = None


def load_model(
    config_path: str,
    checkpoint_path: str,
    tokenizer_path: str,
    device: str = "cuda",
) -> None:
    """Load the model, tokenizer, and initialize the generator.

    Args:
        config_path: Path to model config JSON.
        checkpoint_path: Path to model checkpoint.
        tokenizer_path: Path to tokenizer .model file.
        device: Device to load model on.
    """
    global _generator, _tokenizer, _config

    print(f"Loading model from {checkpoint_path}...")

    # Load config
    _config = PrismConfig.from_json(config_path)

    # Load tokenizer
    _tokenizer = PrismTokenizer(tokenizer_path)

    # Load model
    model = PrismForCausalLM(_config)

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "module" in state_dict:
        state_dict = state_dict["module"]
    model.load_state_dict(state_dict, strict=False)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Initialize generator
    _generator = PrismGenerator(model, _tokenizer, device)

    print(f"Model loaded on {device}. Ready for inference.")


@app.on_event("startup")
async def startup_event():
    """Attempt auto-loading model on startup from environment variables."""
    config_path = os.environ.get("PRISM_CONFIG", "configs/model_10b.yaml")
    checkpoint_path = os.environ.get("PRISM_CHECKPOINT")
    tokenizer_path = os.environ.get("PRISM_TOKENIZER", "tokenizer/prism_tokenizer.model")
    device = os.environ.get("PRISM_DEVICE", "cuda")

    if checkpoint_path and os.path.exists(checkpoint_path) and os.path.exists(tokenizer_path):
        try:
            print(f"Auto-loading model from environment variables on startup...")
            load_model(config_path, checkpoint_path, tokenizer_path, device=device)
        except Exception as e:
            print(f"Warning: Auto-loading model on startup failed: {e}")
    else:
        print("FastAPI server started. Model not pre-loaded. Call load_model() or provide PRISM_CHECKPOINT.")


# ── Endpoints ───────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the model is loaded and ready."""
    return HealthResponse(
        status="healthy" if _generator is not None else "model_not_loaded",
        model_loaded=_generator is not None,
        model_params_billions=_config.num_params_billions if _config else 0.0,
        device=str(_generator.device) if _generator else "none",
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from a prompt."""
    if _generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    # Count prompt tokens
    prompt_tokens = len(_tokenizer.encode(request.prompt, add_bos=True))

    # Generate
    generated_text = _generator.generate(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
    )

    elapsed_ms = (time.time() - start_time) * 1000

    # Count generated tokens
    generated_tokens = len(_tokenizer.encode(generated_text, add_bos=False)) - prompt_tokens

    return GenerateResponse(
        generated_text=generated_text,
        prompt_tokens=prompt_tokens,
        generated_tokens=max(generated_tokens, 0),
        total_time_ms=round(elapsed_ms, 2),
    )
