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

