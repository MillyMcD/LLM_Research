# LLM_Research
This repository contains the Large Language Model research for the MSc AI Dissertation. It requires that you have Docker desktop installed, and a GPU available to you. This project was built using an Intel I7 Predator Triton 300 Acer Laptop with a GeForce RTX3080 (8GB RAM). The fine-tuning was done using Google Colab using an A100 instance in order to export to GGUF for Ollama.

## Installation
To install this project, call
```
docker compose up --build
```

From then on, call
```
docker compose up
```

Navigate to http://localhost:8888 to access the Jupyter Notebook server

## Source Code
All source code is found in the `src/` folder. There are several Python files containing classes and objects to perform this analysis and run

## Notebooks
- `component_experiments.ipynb` : Notebook for running all component experiments i.e. comparing RAG, LLMs and so forth
- `pipeline_experiments.ipynb` : Notebook for testing the pipeline and its hand off mechanisms
- `unsloth_finetuning.ipynb` : Ported directly from Unsloth; for fine-tuning a 4-bit instruct model. NOTE! Exporting to GGUF requires a lot of memory. Recommend copying this notebook and using Colab.
- `compile_results.ipynb` : Notebook for getting results together for presentation

