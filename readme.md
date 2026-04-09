# Adaptive Semantic Parallelism (ASP)

## Overview

This project implements an end-to-end pipeline for processing complex user prompts using structured decomposition, dependency-aware execution, and adaptive model routing.

The system:
- Decomposes prompts into atomic tasks
- Detects dependencies and constructs a DAG
- Executes tasks in parallel or sequentially
- Routes tasks to weak or strong models
- Applies safety checks
- Verifies outputs and re-queries if needed
- Aggregates final response
- Compares against a baseline model

---

## Project Structure
.
├── dataset/Synthetic dataset generation
├── Intent_complexity/ 
├── router/
├── semantic_decom_dependency/ # Decomposition and DAG construction
├── execution_engine/ # Task execution engine
├── safety_system/ # Safety and filtering
├── verification_aggregation/ # Verification, aggregation, feedback
├── config/ # Labels and constants
├── models/ # Saved models
├── evaluation/ # Metrics and results
├── data/ # Generated datasets
├── pipeline.py # Main pipeline runner


---

## Modules

### Dataset Generation
Generates synthetic prompts with:
- Single-task and multi-task queries
- Noise and paraphrasing
- Intent, complexity, and routing labels

### Intent and Complexity Model
- Predicts intent (11 classes)
- Estimates complexity score

### Router
- Predicts weak_model vs strong_model
- Uses complexity and linguistic features

### Decomposer and DAG
- Splits prompt into segments
- Detects dependencies
- Builds DAG using topological sorting

### Execution Engine
- Executes tasks in parallel or sequentially
- Handles retries, fallback, and caching

### Safety System
- Filters unsafe inputs
- Detects hallucination and PII
- Applies guardrails and escalation

### Verification and Aggregation
- Validates outputs per intent
- Computes confidence
- Re-queries weak outputs
- Aggregates final answer

### Pipeline Runner
- Runs full system
- Compares pipeline vs baseline
- Reports latency and cost

---

## Setup

### Create Environment

```bash
conda create -n asp python=3.11 -y
conda activate asp

pip install numpy pandas scikit-learn lightgbm spacy matplotlib
python -m spacy download en_core_web_sm

export GROQ_API_KEY=your_key_here

PYTHONPATH=. python dataset/synthetic_dataset.py --num-samples 5000 --use-llm
PYTHONPATH=. python Intent_complexity/intent_trainer.py --data dataset/dataset.json
PYTHONPATH=. python router/train_router.py --data dataset/dataset.json
python pipeline.py --data data/dataset.jsonl --max-samples 50
python pipeline.py --prompt "Solve x+2=5 and give an image representing it"

