# Reliable Agentic RAG with LLM Trustworthiness Estimates

We will cover following

## Getting Started

Install [uv](https://docs.astral.sh/uv/).

Create a virtual environment using uv

```bash
uv venv --python 3.12
source .venv/bin/activate
```

### Download dataset

We will work with documentation of Nvidia on Triton Inference Server: <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html>

As part of the script below, we will download all the links in html format using `wget`.

```bash
cd scripts
bash get_data.sh
```

### Run data pipeline

```bash
cd ..
uv run agentic_rag/main.py
```
