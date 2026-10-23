# Transformer-Based Spam Detection

A natural language processing project and documentation guide that classifies email and message content as
**spam** or **ham** (not spam). The project fine-tunes and compares three
transformer architectures:

- **BERT** - standard contextual language model
- **RoBERTa** - enhanced BERT-style architecture
- **DistilBERT** - smaller and faster BERT variant

The project includes training notebooks, a Flet desktop interface, preprocessing
logic shared with the trained models, evaluation results, and local model assets.

## Features

- Classifies a message using BERT, RoBERTa, or DistilBERT.
- Preloads the selected model to make repeated predictions faster.
- Uses the same preprocessing and long-message truncation strategy as training.
- Displays the predicted label, confidence, and ham/spam probabilities.
- Includes notebooks for model training, evaluation, and classical ML baselines.

## Project structure

```text
Final_Project_NLP/
├── app_ui/                 # Flet user interface
│   └── README.md           # UI-specific documentation
├── data/                   # Versioned training datasets
├── models/                 # Local fine-tuned model assets (ignored by Git)
├── services/
│   ├── README.md           # Prediction service documentation
│   └── predictor.py        # Model loading, preprocessing, and inference
├── BERT.ipynb              # BERT training and evaluation
├── Distillbert.ipynb       # DistilBERT training and evaluation
├── Roberta.ipynb           # RoBERTa training and evaluation
├── main.py                 # Application entry point
└── requirements.txt        # Python dependencies
```

## Requirements

- Python 3.10 or newer
- A machine with enough memory for transformer inference
- Optional: a CUDA-enabled GPU for training

The application can run on CPU, but model loading and prediction will be slower.

## Installation

Create and activate a virtual environment:

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Model and dataset assets

The datasets in `data/` are included in the repository so that the training
notebooks can be reproduced after cloning. They include the Enron, Ling-Spam,
and SpamAssassin corpora used by the project.

The trained model weights in `models/` are excluded from Git because model files
can be very large. Before running the application, make sure the following model
directories exist:

```text
models/
├── bert_spamassassin/
├── distilbert_spamassassin/
└── roberta_spamassassin/
```

Each model directory must contain at least:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
```

If the model assets are distributed as `models.zip`, extract it at the project
root so that the directories are directly under `models/`, not nested under
another `models/` directory. The training notebooks expect the versioned CSV datasets under `data/`.

## Running the application

From the project root, with the virtual environment activated:

```powershell
python main.py
```

Choose a model, enter or paste a message, and select **Classify Message**.

## Training and evaluation

Open one of the notebooks in Jupyter or VS Code:

```powershell
jupyter notebook
```

The notebooks cover:

1. Loading and combining spam datasets.
2. Text cleaning and preprocessing.
3. Stratified train/validation/test splitting.
4. Transformer fine-tuning.
5. Class-imbalance handling.
6. Metrics such as accuracy, precision, recall, macro F1, and ROC-AUC.
7. Comparison with TF-IDF and classical classifiers.

Training can take significant time and disk space. It may also require additional
configuration for CUDA, depending on the available hardware.

## Inference behavior

The application preprocesses each message by:

- Removing HTML tags.
- Removing common email header lines.
- Converting text to lowercase.
- Removing common boilerplate footers.
- Normalizing whitespace.

Messages longer than the model input limit are truncated by retaining tokens from
both the beginning and end of the message. This preserves potentially useful
context such as opening content and footer links.

## Documentation and contribution notes

This `README.md` is the main project documentation. It explains the project
purpose, installation, dataset and model requirements, application usage,
training workflow, and inference behavior. Keep it updated when setup steps,
dependencies, directory names, or user-facing behavior change.

For more focused documentation, see:

- [`app_ui/README.md`](./app_ui/README.md) for the interface structure and UI
  workflow.
- [`services/README.md`](./services/README.md) for model loading, preprocessing,
  inference, and service errors.

- Commit the CSV files under `data/` when updating or adding training data.
- Do not commit `.env` files, model weights, archives, or Python cache files.
- Keep the preprocessing logic in `services/predictor.py` synchronized with the
  preprocessing used in the training notebooks.
- If model files are hosted externally, document the download and extraction
  process rather than adding large weights directly to normal Git history.
