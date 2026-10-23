from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define a mapping from the UI model names to the directory names
MODEL_MAPPING = {
    "BERT": "bert_spamassassin",
    "DistilBERT": "distilbert_spamassassin",
    "RoBERTa": "roberta_spamassassin",
}

# Base path for models directory
MODELS_BASE_PATH = Path(__file__).parent.parent / "models"
MAX_LEN = 512

# Global cache for models and tokenizers
_model_cache: Dict[str, Any] = {}
_model_load_locks: Dict[str, threading.Lock] = {}
_model_inference_locks: Dict[str, threading.Lock] = {}
_cache_lock = threading.Lock()


class PredictionServiceError(Exception):
    """Base exception for prediction service failures."""


class ModelAssetsNotFoundError(PredictionServiceError):
    """Raised when required model files are missing."""


def preprocess_text(text: str) -> str:
    """
    Must match the exact preprocessing pipeline used during training (notebook Cell 5).
    """
    # Step 1 — Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Step 2 — Remove email header lines
    text = re.sub(
        r'^(From|To|Subject|Date|Cc|Reply-To|Message-ID|'
        r'Return-Path|Received|MIME-Version|Content-Type|'
        r'Content-Transfer-Encoding|X-[\w-]+):.*$',
        '', text, flags=re.MULTILINE | re.IGNORECASE
    )
    # Step 3 — Lowercase
    text = text.lower()
    # Step 4 — Remove boilerplate footers
    text = re.sub(
        r'(this email is confidential|unsubscribe|legal disclaimer'
        r'|this message contains|privileged communication).*',
        '', text, flags=re.IGNORECASE | re.DOTALL
    )
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _truncate_keep_edges(text: str, tokenizer, max_len: int = MAX_LEN) -> str:
    """
    Must match SpamDataset._truncate_keep_edges() from training (notebook Cell 7).
    Keeps the first half + last half of tokens so spam signals at both
    ends of a long message are preserved.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    max_tokens = max_len - 2  # reserve [CLS] and [SEP]
    if len(tokens) <= max_tokens:
        return text
    half = max_tokens // 2
    tokens = tokens[:half] + tokens[-(max_tokens - half):]
    return tokenizer.decode(tokens, skip_special_tokens=True)


@dataclass
class PredictionResult:
    label: str
    confidence: float
    ham_probability: float
    spam_probability: float
    processing_time_ms: int


def _normalize_label(label: str) -> str:
    raw = (label or "").strip().lower()
    if raw in {"ham", "not_spam", "non_spam", "legit", "legitimate", "label_0"}:
        return "ham"
    if raw in {"spam", "junk", "phishing", "label_1"}:
        return "spam"
    return raw or "unknown"


def _validate_model_assets(model_path: Path, model_name: str) -> None:
    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing = [name for name in required_files if not (model_path / name).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise ModelAssetsNotFoundError(
            f"Missing files for {model_name} in {model_path}: {missing_list}"
        )


def _load_model_and_tokenizer(model_name: str) -> None:
    """Loads a model and its tokenizer into the cache if not already present."""
    if model_name in _model_cache:
        return

    # Ensure only one thread loads a given model at a time.
    with _cache_lock:
        model_lock = _model_load_locks.setdefault(model_name, threading.Lock())

    with model_lock:
        if model_name in _model_cache:
            return

        model_dir_name = MODEL_MAPPING.get(model_name)
        if not model_dir_name:
            raise PredictionServiceError(f"Invalid model name: {model_name}")

        model_path = MODELS_BASE_PATH / model_dir_name
        if not model_path.exists():
            raise ModelAssetsNotFoundError(f"Model directory not found: {model_path}")
        _validate_model_assets(model_path, model_name)
    
        print(f"Loading {model_name} model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()  # Set model to evaluation mode
        _model_cache[model_name] = {"model": model, "tokenizer": tokenizer}
        print(f"{model_name} model loaded and cached.")


def _get_model_inference_lock(model_name: str) -> threading.Lock:
    with _cache_lock:
        return _model_inference_locks.setdefault(model_name, threading.Lock())


def preload_model(model_name: str) -> None:
    """Public API used by UI to preload a model without running inference."""
    _load_model_and_tokenizer(model_name)


def get_loaded_models_status() -> Dict[str, bool]:
    """Returns loaded-state for all supported models."""
    return {name: name in _model_cache for name in MODEL_MAPPING}


def predict_message(text: str, model_name: str) -> PredictionResult:
    """
    Classifies a message as spam or ham using the specified model.
    """
    start_time = time.perf_counter()

    if not text or not text.strip():
        raise PredictionServiceError("Input message is empty.")

    # Ensure the model is loaded and cached
    _load_model_and_tokenizer(model_name)

    # Get the model and tokenizer from the cache
    cached_assets = _model_cache[model_name]
    model = cached_assets["model"]
    tokenizer = cached_assets["tokenizer"]

    # Apply text preprocessing before model-specific tokenization.
    processed_text = preprocess_text(text)

    inference_lock = _get_model_inference_lock(model_name)
    with inference_lock:
        # Keep all tokenizer/model operations under one lock; these objects are not thread-safe.
        truncated_text = _truncate_keep_edges(processed_text, tokenizer)
        inputs = tokenizer(
            truncated_text,
            return_tensors="pt",
            max_length=MAX_LEN,
            truncation=True,
            padding=True,
        )

        with torch.no_grad():
            logits = model(**inputs).logits

    # Calculate probabilities and get prediction
    probabilities = torch.softmax(logits, dim=1).squeeze()
    ham_probability = probabilities[0].item()
    spam_probability = probabilities[1].item()
    prediction = torch.argmax(probabilities)
    confidence = probabilities[prediction].item()
    raw_label = model.config.id2label[prediction.item()]
    label = _normalize_label(raw_label)

    end_time = time.perf_counter()
    processing_time_ms = int((end_time - start_time) * 1000)

    return PredictionResult(
        label=label,
        confidence=confidence,
        ham_probability=ham_probability,
        spam_probability=spam_probability,
        processing_time_ms=processing_time_ms,
    )
