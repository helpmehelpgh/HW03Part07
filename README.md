# CPE 487/587 – HW1: Binary Classification with Gradient Descent (PyTorch)

This project implements a gradient-descent based binary classifier using PyTorch autograd.
It provides a function `binary_classification` and a demo script that trains the model and saves
a loss-vs-epochs plot as a timestamped PDF file.

## Project Structure

- `src/mchnpkg/deepl/two_layer_binary_classification.py`  
  Contains the function `binary_classification(...)`
- `src/mchnpkg/__init__.py` and `src/mchnpkg/deepl/__init__.py`  
  Package initialization for imports
- `scripts/binaryclassification_impl.py`  
  Demonstrates training and generates the loss plot PDF

## Requirements

- Python 3.11+ (recommended: Python 3.12)
- `uv` package manager

## Install (UV)

From the project root:

```bash
# create virtual environment
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate

# install dependencies from pyproject.toml
uv sync
