
# Final Project

> **Apple Silicon Mac:** use a native arm64 Python build (e.g. `CONDA_SUBDIR=osx-arm64 conda create ...`). Linux, Windows, and Intel Mac work with a standard Python install.

## Setup

### 1. Create and activate a conda environment

```bash
conda create -n project python=3.10
conda activate project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

Edit the flags at the top of [main.py](main.py) to control which steps execute:

| Flag           | Default | Description                                                  |
|----------------|---------|--------------------------------------------------------------|
| `PREP_DATASET` | `True`  | Download dataset from Kaggle and prepare train/val/test splits |
| `TRAIN_CNN`    | `True`  | Train the CNN model (saves to `cnn_model.pth`)               |
| `TRAIN_DNN`    | `True`  | Train the DNN model (saves to `dnn_model.pth`)               |
| `TEST_CNN`     | `True`  | Evaluate the CNN on the test set                             |
| `TEST_DNN`     | `True`  | Evaluate the DNN on the test set                             |

Set any flag to `False` to skip that step. If `TRAIN_CNN`/`TRAIN_DNN` is `False` but `TEST_CNN`/`TEST_DNN` is `True`, the saved `.pth` file will be loaded automatically.

Then run:

```bash
python main.py
```
