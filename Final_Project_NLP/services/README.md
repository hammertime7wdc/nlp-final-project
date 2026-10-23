# Prediction Services

This directory contains the application logic used to load transformer models
and classify messages. It provides the boundary between the Flet interface and
the local BERT, RoBERTa, and DistilBERT model assets.

## Files

- [`predictor.py`](./predictor.py) - Loads and caches models, preprocesses
  messages, runs inference, validates model assets, and returns prediction
  results.
- [`__init__.py`](./__init__.py) - Marks this directory as a Python package.

## Supported models

The service maps user-facing model names to directories under `models/`:

| UI name | Model directory |
| --- | --- |
| `BERT` | `models/bert_spamassassin/` |
| `DistilBERT` | `models/distilbert_spamassassin/` |
| `RoBERTa` | `models/roberta_spamassassin/` |

Each model directory must contain the model configuration, weights, and
tokenizer files required by Hugging Face Transformers. See the main
[`README.md`](../README.md) for the complete asset requirements.

## Public operations

- `preload_model(model_name)` - Loads and caches a selected model without
  running a prediction.
- `get_loaded_models_status()` - Reports which supported models are currently
  loaded.
- `predict_message(text, model_name)` - Preprocesses and classifies a message,
  returning a `PredictionResult`.

## Prediction pipeline

Before inference, [`predictor.py`](./predictor.py):

1. Rejects empty input.
2. Loads the requested model and tokenizer if they are not already cached.
3. Removes HTML tags and common email headers.
4. Converts the message to lowercase and removes boilerplate footers.
5. Normalizes whitespace.
6. Preserves the beginning and end of messages that exceed the token limit.
7. Runs the transformer model and calculates ham/spam probabilities.

Models and tokenizers are cached after the first load. Separate locks protect
model loading and inference so the service can be used safely by asynchronous
UI actions.

## Error handling

The service raises `PredictionServiceError` for invalid input, unsupported
models, and prediction failures related to the application contract.
`ModelAssetsNotFoundError` specifically identifies missing model directories or
required files. The UI is responsible for presenting these errors to users.

## Maintenance notes

Keep the preprocessing implementation synchronized with the preprocessing used
in the training notebooks. Changing one without the other can reduce prediction
quality because the deployed model will receive differently formatted text.

