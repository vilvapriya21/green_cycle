# 🌱 Green-Cycle – Autonomous Waste Auditor

An intelligent smart-city waste auditing system that classifies waste descriptions and generates city-compliant disposal plans using Machine Learning and an AI Agent.

---

# 📌 Project Overview

Green-Cycle simulates a smart city application where citizens submit waste descriptions via text.

The system:

1. Classifies the waste into one of:

   * Hazardous
   * Recyclable
   * Compost

2. Uses an AI Agent with few-shot prompting to generate a disposal plan based on simulated city policy.

3. Exposes the functionality through a FastAPI REST API.

4. Is fully containerised using Docker.

---

# 🧠 1. Machine Learning Core

## Model Choice

We implemented a **Logistic Regression classifier** using Scikit-learn.

### Why Logistic Regression?

* Suitable for text classification problems
* Works well with sparse TF-IDF vectors
* Provides probability estimates
* Less sensitive to noise compared to KNN
* Computationally efficient

---

## Preprocessing

Text preprocessing is performed using **spaCy** and includes:

* Lowercasing
* Lemmatization (NOT stemming)
* Stop-word removal
* Punctuation removal

This ensures semantic consistency (e.g., “batteries” → “battery”).

---

## Feature Engineering

* TF-IDF Vectorizer used to convert text into numerical features.

---

## Training & Evaluation

* Dataset size: 100+ examples
* Balanced across 3 classes
* Train/Test Split: 80/20
* Cross-validation used to validate stability

### Results

* Training Accuracy: 0.9688
* Test Accuracy: 0.7917
* Cross Validation Score: 0.7667

### Overfitting Check

Overfitting was evaluated by comparing:

* Training vs Test accuracy
* Cross-validation mean score

Training Accuracy: 96.88%  
Test Accuracy: 79.17%  
Cross-Validation Mean Accuracy: 76.67%

There is a moderate gap between training and test accuracy (~17%).

This suggests mild overfitting, which is expected given:

- The relatively small dataset (~100 samples)
- Sparse TF-IDF features
- Limited domain vocabulary

However, cross-validation scores are consistent (0.70–0.79), 
indicating that the model generalises reasonably well 
and does not collapse on unseen data.

To mitigate overfitting:
- Regularised Logistic Regression was used
- Balanced dataset across classes
- Cross-validation was performed

Given the small dataset size, the model performance is considered acceptable and stable.

### Classification Report (Test Set)

| Class       | Precision | Recall | F1-score |
|------------|-----------|--------|----------|
| Compost     | 0.62      | 1.00   | 0.76     |
| Hazardous   | 1.00      | 0.62   | 0.77     |
| Recyclable  | 1.00      | 0.75   | 0.86     |

Overall Accuracy: 79.17%
Macro Avg F1: 0.80

---

# 🤖 2. AI Agent

## Architecture

The AI Agent performs:

1. Logic-based decision:

   * If Hazardous → check special handling rules
   * If Recyclable → check recycling bin policy
   * If Compost → check compost guidelines

2. Fetch simulated City Policy

3. Construct a Few-Shot Prompt

4. Call External LLM API (OpenAI / HuggingFace)

5. Return final disposal plan

---

## Few-Shot Prompt Design

The agent uses structured examples:

You are a waste disposal assistant...

Examples:
Description: empty plastic bottle
Category: Recyclable
City Policy: Plastic bottles must be rinsed and placed in the blue bin.
Disposal Plan: Rinse and place in the blue bin.

...

Then processes:
Description: {description}
Category: {category}
City Policy: {policy}
Disposal Plan:

---

## API Security

* API keys are stored in environment variables:

  * OPENAI_API_KEY
  * HF_TOKEN
* No hardcoded secrets
* Graceful error handling implemented:

  * Network failure
  * Rate limits
  * Invalid response

If API fails → fallback rule-based disposal plan is returned.

---

# 🌐 3. API Design

Built using FastAPI.

## Endpoints

### POST /classify

Request:
{
"text": "used batteries"
}

Response:
{
"category": "Hazardous",
"confidence": 0.84
}

---

### POST /disposal

Request:
{
"text": "used batteries"
}

Response:
{
"category": "Hazardous",
"confidence": 0.84,
"disposal_plan": "Seal in leak-proof container and take to hazardous waste facility."
}

---

## Validation

* Pydantic models used
* Proper HTTP status codes:

  * 200 – Success
  * 400 – Invalid input
  * 500 – Server error

---

# 🐳 4. Docker Deployment

## Build Image

docker build -t green-cycle .

## Run Container

docker run -p 8000:8000 -e OPENAI_API_KEY=your_key green-cycle

## Optimisations

* Base image: python:3.10-slim
* Minimal layer size
* Cleared pip cache
* Environment variables passed at runtime

---

# 📂 Project Structure

```
# 📂 Project Structure

green_cycle/
│
├── app/
│   ├── main.py                  # FastAPI entry point
│   │
│   ├── api/                     # HTTP layer only
│   │   └── routes.py
│   │
│   ├── core/
│   │   └── logging_config.py
│   │
│   ├── services/                # Orchestration layer
│   │   └── waste_audit_service.py
│   │
│   ├── ml/
│   │   ├── preprocessor.py      # spaCy cleaning
│   │   ├── classifier.py        # Model loading & prediction
│   │   └── train.py             # Training & evaluation
│   │
│   ├── agent/
│   │   ├── policy.py
│   │   ├── prompt_builder.py
│   │   └── llm_client.py
│   │
│   ├── schemas/
│   │   └── models.py            # Pydantic models
│   │
│   └── config.py                # Centralised settings
│
├── models/
│   ├── classifier.joblib
│   └── vectorizer.joblib
│
├── data/
│   ├── waste_data.csv
│   └── generate_dataset.py
│
├── tests/
│   ├── test_classifier.py
│   ├── test_agent.py
│   └── test_routes.py
│
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

# ⚙️ Local Setup

1. Create virtual environment
2. Install dependencies:
   pip install -r requirements.txt
3. Run:
   uvicorn app.main:app --reload

---
# 🔐 Environment Variables

Create a `.env` file (not committed to Git):

OPENAI_API_KEY=your_openai_key
HF_TOKEN=your_huggingface_token

These keys are injected at runtime and are never hardcoded.

---

# 🧪 Example CURL

curl -X POST "http://127.0.0.1:8000/disposal" 
-H "Content-Type: application/json" 
-d '{"text": "old paint can"}'

---

# 🧹 Code Hygiene

* PEP8 naming conventions
* Modular structure
* Docstrings for functions/classes
* Clear separation of ML, Agent, API, Config

---

# 📊 Design Choices Summary

* Logistic Regression selected for stability
* TF-IDF for robust text vectorisation
* spaCy for linguistic-level preprocessing
* Few-shot prompting to demonstrate agent reasoning
* Dockerised for portability

---

# 👩‍💻 Author

Vilva Priya K
