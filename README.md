
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