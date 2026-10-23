# Application UI

This directory contains the Flet user interface for the spam detection
application. It is responsible for presenting the application window, accepting
message input, allowing the user to select a model, and displaying prediction
results and errors.

## Files

- [`main_view.py`](./main_view.py) - Builds the main application page, model
  selection cards, message input, classify action, loading state, and error
  handling.
- [`result_panel.py`](./result_panel.py) - Displays classification results,
  confidence, probabilities, and processing time.
- [`theme/palette.py`](./theme/palette.py) - Defines shared colors and visual
  theme constants.
- [`__init__.py`](./__init__.py) - Exposes the main page function.

## UI flow

1. The user selects BERT, RoBERTa, or DistilBERT.
2. The selected model is preloaded asynchronously through the services layer.
3. The user enters a message.
4. The UI sends the message and selected model to the prediction service.
5. The result panel displays the predicted label and probability values.
6. Any expected service errors are shown in the application error banner.

## Running the UI

Start the application from the project root:

```powershell
python main.py
```

The entry point in [`main.py`](../main.py) imports the `main` page function from
[`main_view.py`](./main_view.py) and starts Flet.

## UI and service boundary

The UI should handle presentation and user interaction only. Model loading,
text preprocessing, inference, and prediction result calculation belong in
[`services/`](../services/). This keeps the interface independent from the
underlying transformer implementation.

