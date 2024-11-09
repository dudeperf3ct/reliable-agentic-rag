# Reliable Agentic RAG with LLM Trustworthiness Estimates

We will cover following

## Getting Started

Install [uv](https://docs.astral.sh/uv/).

Create a virtual environment using `uv`

```bash
uv venv --python 3.12
source .venv/bin/activate
```

Install the dependencies required to run this project inside the virtual environment.

```bash
uv pip install -r pyproject.toml
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

pwd
/home/username/reliable-agentic-rag
```

> [!Note]
> Make sure you are running the following command from the root of the project (inside `reliable-agentic-rag` folder).

```bash
python agentic_rag/run.py data
```

To know more about data pipeline, refer to the documentation on [Datapipeline](./docs/Datapipeline.md).

> [!TIP]
> The documentation for parameters that can be configuration as part of data pipeline [here](./docs/Datapipeline.md#configuration).

### Run query pipeline

> [!Note]
> Make sure you are running the following command from the root of the project (inside `reliable-agentic-rag` folder).

```bash
python agentic_rag/run.py query --query-text "How to make custom layers of TensorRT work in Triton?"
```
