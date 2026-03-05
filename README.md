# Green-Cycle: Autonomous Waste Auditor

## Project Overview

**Green-Cycle** is an intelligent waste auditing system that classifies waste descriptions and generates disposal instructions according to simulated city waste policies.

The system demonstrates an **end-to-end AI engineering pipeline** combining:

* **Machine Learning** for waste classification
* **AI Agent (LLM-based)** for disposal planning
* **FastAPI REST API** for serving predictions
* **Docker containerization** for reproducible deployment

The goal is to simulate a **smart city assistant** that allows citizens to report waste items through text and receive guidance on how to dispose of them properly.

This project was implemented following the requirements of the **“Green-Cycle Autonomous Waste Auditor” assessment**, which requires an ML model, an LLM agent, a REST API service, and Docker deployment. 

---

# System Workflow

1. User sends a **text description of a waste item**
2. Text is **cleaned and normalized using NLP preprocessing**
3. The **ML classifier predicts the waste category**
4. The **AI agent retrieves the relevant city policy**
5. The **LLM generates a disposal plan**
6. The system returns **classification and disposal instructions via API**

---

# Waste Categories

The ML classifier predicts one of the following classes:

| Category   | Description                               |
| ---------- | ----------------------------------------- |
| Recyclable | Materials that can be reused or processed |
| Compost    | Organic waste that can decompose          |
| Hazardous  | Materials requiring special handling      |

---

# System Architecture

```
User Request
     |
     v
FastAPI Endpoint
     |
     v
ML Classification Pipeline
(Text preprocessing + TF-IDF + Logistic Regression)
     |
     v
Waste Category
     |
     v
AI Agent
(Few-shot prompting + policy rules)
     |
     v
Disposal Plan
     |
     v
JSON API Response
```

---

# Project Structure

```
green_cycle/
│
├── app/
│   ├── agent/               # LLM interaction and prompting
│   │   ├── llm_client.py
│   │   ├── policy.py
│   │   └── prompt_builder.py
│   │
│   ├── api/                 # FastAPI routes
│   │   └── routes.py
│   │
│   ├── ml/                  # Machine learning pipeline
│   │   ├── classifier.py
│   │   ├── preprocessor.py
│   │   └── train.py
│   │
│   ├── schemas/             # Pydantic request/response models
│   │
│   ├── services/            # Business logic layer
│   │   └── waste_audit_service.py
│   │
│   ├── config.py
│   └── main.py              # FastAPI entrypoint
│
├── data/
│   ├── waste_data.csv
│   └── generate_dataset.py
│
├── notebooks/
│   ├── exploration.ipynb
│   └── green_cycle_ml.ipynb
│
├── tests/
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Machine Learning Component

The ML component classifies waste descriptions such as:

* "banana peel"
* "empty plastic bottle"
* "used batteries"
* "paint thinner can"

The output is one of the three waste categories.

---

# Text Preprocessing

The preprocessing pipeline performs the following NLP steps:

* Lowercasing
* Lemmatization
* Stopword removal
* Punctuation cleaning

Libraries used:

* **spaCy**
* **NLTK**

These steps normalize the text before feature extraction.

---

# Feature Engineering

Text is converted into numerical features using **TF-IDF Vectorization**.

TF-IDF helps represent the importance of words within waste descriptions relative to the entire dataset.

---

# Model Selection

The classifier uses **Logistic Regression (scikit-learn)**.

Reasons for selecting Logistic Regression:

* Effective for text classification
* Works well with TF-IDF features
* Efficient and lightweight
* Suitable for real-time inference in APIs

---

# Training Configuration

| Parameter        | Value               |
| ---------------- | ------------------- |
| Dataset Size     | 301 samples         |
| Classes          | 3                   |
| Train/Test Split | 80/20               |
| Feature Type     | TF-IDF              |
| Model            | Logistic Regression |

---

# Model Evaluation

Evaluation results are shown in **green_cycle_ml.ipynb**.

| Metric                | Result |
| --------------------- | ------ |
| Training Accuracy     | ~0.95  |
| Test Accuracy         | ~0.92  |
| Cross-Validation Mean | ~0.91  |

Cross-validation results are consistent with test performance, indicating that the model generalizes well to unseen data.

---

# AI Agent

The AI agent generates a **human-readable disposal plan** using:

* Waste description
* Predicted waste category
* City waste policy rules

---

# City Policy Simulation

City policies are implemented using a simple rule dictionary.

| Category   | Policy                                           |
| ---------- | ------------------------------------------------ |
| Recyclable | Rinse items and place in recycling bins          |
| Compost    | Dispose in organic waste containers              |
| Hazardous  | Deliver to hazardous waste collection facilities |

---

# Agent Decision Logic

Before generating the prompt, the agent determines the appropriate policy:

```
if category == "Hazardous":
    apply hazardous waste handling rules
elif category == "Recyclable":
    apply recycling policy
else:
    apply composting policy
```

This step demonstrates **logic-based decision making before invoking the LLM**.

---

# Few-Shot Prompting

Example prompt used for the LLM:

```
You are a waste disposal assistant.

Example 1:
Description: empty plastic bottle
Category: Recyclable
City Policy: Plastic bottles must be rinsed and placed in recycling bins.
Disposal Plan: Rinse the bottle and place it in the recycling bin.

Example 2:
Description: used batteries
Category: Hazardous
City Policy: Batteries must be taken to a hazardous waste facility.
Disposal Plan: Take the batteries to the hazardous waste collection center.

Now process:

Description: {description}
Category: {category}
City Policy: {policy}
Disposal Plan:
```

---

# LLM Integration

The system uses an **OpenAI-compatible chat completion API**.

By default, the project is configured to use **Groq LLM inference**.

Environment variables control the LLM configuration.

---

# Environment Variables

Create a `.env` file in the project root.

```
# LLM Provider Configuration

LLM_API_KEY=your_api_key

LLM_MODEL=llama-3.1-8b-instant

LLM_BASE_URL=https://api.groq.com/openai/v1/chat/completions

LLM_TIMEOUT=15

LLM_TEMPERATURE=0.3
```

Because the system uses an **OpenAI-compatible API interface**, the same configuration can be adapted to other providers by modifying the model name and base URL.

---

# REST API

Two REST endpoints are provided using **FastAPI**.

---

## POST /classify

Predicts the waste category.

Request

```
{
"text": "banana peel"
}
```

Response

```
{
"category": "Compost"
}
```

---

## POST /disposal

Returns the waste category and disposal plan.

Request

```
{
"text": "used batteries"
}
```

Response

```
{
"category": "Hazardous",
"plan": "Take the batteries to the hazardous waste collection facility."
}
```

---

# Running the Project Locally

## Install Dependencies

```
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create a `.env` file using the environment variables described earlier.

---

## Run FastAPI Server

```
uvicorn app.main:app --reload
```

Server runs at:

```
http://localhost:8000
```

Interactive API documentation:

```
http://localhost:8000/docs
```

---

# Example API Requests

## Classify Waste

```
curl -X POST http://localhost:8000/classify \
-H "Content-Type: application/json" \
-d '{"text":"glass bottle"}'
```

---

## Generate Disposal Plan

```
curl -X POST http://localhost:8000/disposal \
-H "Content-Type: application/json" \
-d '{"text":"paint thinner can"}'
```

---

# Docker Deployment

The application is containerized using Docker for consistent deployment.

---

## Dockerfile Overview

Key features:

* Uses **python:3.10-slim** base image
* Installs dependencies from **requirements.txt**
* Copies application code into container
* Runs FastAPI server using **uvicorn**

Example Dockerfile structure:

```
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
```

---

## Build Docker Image

```
docker build -t green-cycle .
```

---

## Run Container

```
docker run -p 8000:8000 \
-e LLM_API_KEY=your_api_key \
green-cycle
```

API will be accessible at:

```
http://localhost:8000
```

---

# Testing

Run unit tests using:

```
pytest tests/
```

Tests cover:

* ML classifier functionality
* AI agent behavior
* API endpoints
* Service layer logic

---

# Security

API keys are **never stored in the repository**.

Secrets are injected using **environment variables** during runtime.

---

# Future Improvements

Possible enhancements include:

* Image-based waste classification
* Expanded waste categories
* Retrieval-Augmented Generation for policy retrieval
* Integration with real municipal waste APIs
* Cloud deployment using Kubernetes

---

# Author

K Vilva Priya
AI/ML Engineering Project
