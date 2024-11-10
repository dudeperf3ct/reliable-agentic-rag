# Reliable Agentic RAG with LLM Trustworthiness Estimates

This project is an exploration to understand RAG agents and attempt to replicate the blog titled "Reliable Agentic RAG with LLM Trustworthiness Estimates" : <https://pub.towardsai.net/reliable-agentic-rag-with-llm-trustworthiness-estimates-c488fb1bd116>

Tested this on machine with following configuration

```txt
Python - 3.12
uv - 0.4.25
GPU - Nvidia GeForce RTX 3060 Mobile
OS - Ubuntu 22.04.5 LTS
```

## Getting Started

Install [uv](https://docs.astral.sh/uv/).

Create a virtual environment using `uv`

```bash
uv venv --python 3.12
source .venv/bin/activate
```

Install the dependencies required to run this project inside the virtual environment.

```bash
uv sync
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

Running this command creates a `milvus.db` file that acts as a knowledge base. To know more about data pipeline, refer to the documentation on [Datapipeline](./docs/Datapipeline.md).

> [!TIP]
> The documentation for parameters that can be configuration as part of data pipeline [here](./docs/Datapipeline.md#configuration).

### Run query pipeline

```bash
python agentic_rag/run.py query --query-text "How to make custom layers of TensorRT work in Triton?"
```

> [!WARNING]
> One (or two) API key(s) should be added to the `.env` file.<br><br>
> Trustworthy Language Model (TLM) by cleanlab.ai to estimate trustworthy score. Get API key from here: <https://app.cleanlab.ai/account> after creating an account.<br><br>
> API key for an LLM is required to be added in the `.env` file.
> If LLM is hosted locally, no API key is required. Configure only `LLM_MODEL` and `LLM_API_BASE` parameters.<br>
> If LLM is closed-source, API key is required to be added in the `.env` file.<br>

For more information on how to configure various LLM providers, refer the [documentation](./docs/Querypipeline.md#llm).

> [!TIP]
> The documentation for parameters that can be configuration as part of query pipeline [here](./docs/Querypipeline.md#configuration).

## Documentation

### Data pipline

Documentation: [docs](./docs/Datapipeline.md)

### Query pipline

Documentation: [docs](./docs/Querypipeline.md)

## Agentic RAG

Strictly, the approach implemented as part of this project is <u>not an agentic RAG approach</u>. We are manually providing the list of retrieval strategies, calling the necessary functions for the corresponding strategy and selecting the next strategy depending on the trustworthiness score. To implement a truly autonomus agentic RAG approach, one approach outlined in [this blog](https://vectorize.io/how-i-finally-got-agentic-rag-to-work-right/) is to use JSON mode for structred responses or creating multiple agents to collaborate.

Some questions of my own

- Will this fully autonomous agentic RAG approach outperform the current semi-automated RAG approach? What are advantages of using one over the other?
- In current RAG approach, what different approaches that can be used to replace the [uncertainity estimator](./docs/Querypipeline.md#uncertainity-estimator) component?
- Is this RAG approach reliable and robust to all scenarios?

## Recommended Readings

- <https://pub.towardsai.net/reliable-agentic-rag-with-llm-trustworthiness-estimates-c488fb1bd116>
- <https://vectorize.io/how-i-finally-got-agentic-rag-to-work-right/>

## Further work

- [ ] Implement a [agentic RAG processing loop](https://vectorize.io/how-i-finally-got-agentic-rag-to-work-right/) (explained in Agentic RAG processing loop section of the link)
- [ ] Make configuration more intutive (using `pydantic`)
