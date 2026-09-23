from __future__ import annotations

from typing import Literal

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer


RerankBackend = Literal["torch", "onnx", "openvino"]


def load_embedding_model(model_name: str, *, device: str, max_seq_length: int) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    return model


def load_rerank_model(
    model_name: str,
    *,
    device: str,
    max_length: int,
    backend: RerankBackend = "torch",
) -> CrossEncoder:
    kwargs = {
        "device": device,
        "max_length": max_length,
        "activation_fn": torch.nn.Sigmoid(),
    }
    if backend != "torch":
        # SentenceTransformers v5 delegates optimized runtimes through its backend
        # abstraction. The optional runtime dependencies are installed only when the
        # inference image is built with INSTALL_OPTIMIZED_RERANK=1.
        kwargs["backend"] = backend
    return CrossEncoder(model_name, **kwargs)
