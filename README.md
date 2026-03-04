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

## Dataset

## Dataset

The dataset contains 301 synthetic and manually curated examples.

Class distribution:
- Recyclable: 101
- Compost: 100
- Hazardous: 100

To improve real-world robustness and reduce prediction uncertainty,
the dataset was expanded with diverse sentence-level variations,
context phrases, and modifier combinations.

This increased vocabulary diversity and improved generalisation
across natural-language inputs.

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

- Dataset size: 301 samples
- Balanced across 3 classes
- Train/Test Split: 80/20
- Cross-validation (5-fold) used to validate stability

### Results

- Training Accuracy: 94.17%
- Test Accuracy: 83.61%
- Cross-Validation Mean Accuracy: 85.37%
- Cross-Validation Std: 0.0554

## Overfitting Analysis

Training Accuracy: 94.17%  
Test Accuracy: 83.61%  
Generalisation Gap: ~10.5%

The gap between training and test accuracy is moderate and acceptable
for a TF-IDF based text classification model.

Compared to the earlier smaller dataset (~100 samples), the expanded dataset:

- Reduced overfitting
- Increased cross-validation stability
- Improved macro F1 score
- Reduced prediction uncertainty on sentence-style inputs

The model demonstrates strong generalisation with consistent
cross-validation performance (0.78 – 0.93 across folds).

### Classification Report (Test Set)

| Class       | Precision | Recall | F1-score |
|------------|-----------|--------|----------|
| Compost     | 0.72      | 0.90   | 0.80     |
| Hazardous   | 0.93      | 0.70   | 0.80     |
| Recyclable  | 0.90      | 0.90   | 0.90     |

Overall Accuracy: 83.61%  
Macro Avg F1: 0.83

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
