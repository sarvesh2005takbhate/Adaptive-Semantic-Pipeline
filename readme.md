# Adaptive Semantic Decomposition & Parallelism (ASP)

> **An Intelligent Pipeline for Cost-Efficient Multi-Model LLM Routing via Semantic DAG Decomposition**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Theory & Background](#theory--background)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Detailed Usage](#detailed-usage)
8. [Dataset Generation](#dataset-generation)
9. [Model Training](#model-training)
10. [Pipeline Evaluation](#pipeline-evaluation)
11. [Interactive Mode](#interactive-mode)
12. [Configuration](#configuration)
13. [Results](#results)
14. [Troubleshooting](#troubleshooting)

---

## Project Overview

Modern LLM applications face a critical trade-off: **strong models** (GPT-4, Llama 70B) deliver high-quality reasoning but incur significant latency and cost, while **weak models** (Llama 8B, Gemma) are fast and cheap but struggle with complex tasks.

**ASP** solves this by:
1. **Decomposing** complex user prompts into a Directed Acyclic Graph (DAG) of semantically meaningful sub-tasks
2. **Classifying** each sub-task's intent and complexity using a fine-tuned DistilBERT multi-task learning (MTL) model
3. **Routing** each sub-task to the most cost-effective model (strong vs. weak) using a trained LightGBM classifier
4. **Executing** independent sub-tasks in parallel while respecting dependency chains
5. **Verifying** outputs using NLI-based consistency checks and LLM-as-a-Judge quality assessment
6. **Aggregating** results into a coherent final answer

### Key Innovation

Unlike simple prompt routing (which routes an entire prompt to one model), ASP performs **segment-level routing**: a single user prompt like *"Solve this integral and then explain the result in simple terms"* is decomposed into `[math → strong_model]` and `[explanation → weak_model]`, with the explanation segment depending on the math segment's output.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER PROMPT                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  SAFETY PRE │◄── Toxicity + Injection + PII
                    │   CHECK     │    (3-layer, hash-cached)
                    └──────┬──────┘
                           │
               ┌───────────▼───────────┐
               │  SEMANTIC DECOMPOSER  │◄── spaCy + SentenceTransformer
               │  + DAG BUILDER        │    + Constituency Parsing
               │  + DependencyAnalyzer │    + Critical Path Analysis
               └───────────┬───────────┘
                           │
                ┌──────────▼──────────┐
                │  INTENT & COMPLEXITY │◄── Fine-tuned DistilBERT MTL
                │  ESTIMATOR           │    (11 intents + regression)
                │  + OOD Detection     │    + Attention Entropy
                └──────────┬──────────┘
                           │
                   ┌───────▼───────┐
                   │ SMART ROUTER  │◄── LightGBM + Heuristic Hybrid
                   │ + Strong-     │    + Strong-Inheritance Propagation
                   │  Inheritance  │
                   └───────┬───────┘
                           │
            ┌──────────────▼──────────────┐
            │     DAG EXECUTION ENGINE    │
            │  ┌─────────┐ ┌───────────┐ │
            │  │Parallel  │ │Speculative│ │◄── Async workers
            │  │Execution │ │ Executor  │ │    + Semantic cache
            │  └────┬─────┘ └─────┬─────┘ │    + Context Blackboard
            │       │             │        │
            │  ┌────▼─────────────▼─────┐  │
            │  │  Context Blackboard    │  │◄── Token-efficient
            │  │  (Distilled Insights)  │  │    context passing
            │  └────────────────────────┘  │
            └──────────────┬──────────────┘
                           │
                ┌──────────▼──────────┐
                │  QUALITY JUDGE      │◄── LLM-as-a-Judge
                │  (Weak → Strong     │    Self-check escalation
                │   Escalation)       │
                └──────────┬──────────┘
                           │
                    ┌──────▼──────┐
                    │ SAFETY POST │◄── Output scanning
                    │   CHECK     │    + PII redaction
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │ VERIFICATION &          │◄── NLI cross-encoder
              │ AGGREGATION             │    + AST/semantic checks
              │ (Coherence Checking)    │    + Intelligent merging
              └─────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │ FINAL ANSWER│
                    └─────────────┘
```

### Model Registry (Eager Loading)

All ML models are loaded into a singleton `ModelRegistry` at startup (~22s on GPU):

| Model | Purpose | Size | Device |
|-------|---------|------|--------|
| DistilBERT MTL | Intent classification + Complexity regression | ~66M params | CUDA/CPU |
| LightGBM | Binary routing (strong/weak) | ~50KB | CPU |
| SentenceTransformer (all-MiniLM-L6-v2) | Semantic similarity | ~22M params | CUDA/CPU |
| spaCy (en_core_web_sm) + benepar | Syntactic decomposition | ~12M params | CPU |
| Toxicity Classifier | Safety input scanning | ~66M params | CUDA/CPU |
| Injection Detector | Prompt injection detection | ~125M params | CUDA/CPU |
| NLI Cross-Encoder (DeBERTa-v3-small) | Verification consistency | ~44M params | CUDA/CPU |
| Presidio / Regex | PII detection & redaction | Regex fallback | CPU |

---

## Directory Structure

```
final_project/
├── pipeline.py                    # Main orchestrator (8-stage pipeline)
├── model_registry.py              # Singleton eager model loader
├── run_interactive.py             # Interactive runner with API key mgmt
├── manual.py                      # Manual testing utilities
├── .env                           # API keys (GROQ, GEMINI, etc.)
├── requirements.txt               # Python dependencies
│
├── config/
│   └── labels.py                  # Intent labels, strong/weak sets, route constants
│
├── dataset/
│   ├── synthetic_dataset.py       # Synthetic dataset generator (10K prompts)
│   ├── dataset.json               # Generated dataset (JSONL format)
│   └── dataset_stats.json         # Dataset statistics
│
├── Intent_complexity/
│   ├── intent_estimator.py        # DistilBERT MTL inference (intent + complexity)
│   └── intent_trainer.py          # MTL training script (fine-tuning)
│
├── router/
│   ├── router.py                  # LightGBM routing + strong-inheritance
│   └── train_router.py            # Router training + evaluation
│
├── semantic_decom_dependency/
│   └── decomposition_dependency.py # Semantic decomposer + DAG builder
│                                    # + DependencyAnalyzer + ContextFencer
│
├── execution_engine/
│   ├── execution_engine.py        # DAG-aware async executor
│   │                              # + SpeculativeExecutor + SemanticCache
│   └── blackboard.py              # Context Blackboard + ContextDistiller
│
├── verification_aggregation/
│   ├── aggregator.py              # NLI verification + intelligent merging
│   └── quality_judge.py           # LLM-as-a-Judge quality assessment
│
├── safety_system/
│   └── safety_system.py           # 3-layer safety (regex + ML + PII)
│
├── models/
│   ├── intent_complexity_mtl/     # Trained DistilBERT MTL checkpoint
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── tokenizer files
│   ├── router_model.txt           # Trained LightGBM booster
│   └── router_config.json         # Router feature configuration
│
└── evaluation/
    ├── pipeline_results.json      # Evaluation metrics
    ├── detailed_outputs.json      # Per-prompt outputs (first 30)
    ├── mtl_results.json           # MTL training metrics
    ├── router_results.json        # Router training metrics
    └── *.png                      # Evaluation plots
```

---

## Theory & Background

### 1. Semantic Decomposition

Complex prompts often contain multiple implicit sub-tasks. We use a hybrid approach:

- **spaCy NLP**: Sentence segmentation, dependency parsing, and named entity recognition
- **SentenceTransformer**: Semantic similarity between segments to detect related tasks
- **Constituency Parsing (benepar)**: Grammatical structure analysis for precise splitting points
- **Context Fencing**: Code blocks (`\`\`\`...\`\`\``) and math expressions ($...$) are protected from splitting

### 2. DAG Construction & Critical Path Analysis

Segments are organized into a Directed Acyclic Graph based on:
- **Explicit dependency markers**: "then", "based on", "using the result"
- **Coreference resolution**: Pronouns referring to earlier segments
- **Semantic dependency**: When one task's output is logically needed by another

The `DependencyAnalyzer` computes:
- **Critical path**: The longest dependency chain (determines minimum latency)
- **Parallelism ratio**: Width-to-depth ratio (higher = more parallelizable)
- **Priority ordering**: Segments with most descendants execute first

### 3. Multi-Task Learning (MTL) for Intent & Complexity

A single DistilBERT model with dual heads:
- **Intent Head**: 11-class classification (math, code, simulation, research, prediction, data_analysis, translation, summarization, explanation, communication, documentation)
- **Complexity Head**: Regression [0,1] for task difficulty estimation
- **OOD Detection**: Attention entropy analysis to detect out-of-distribution inputs

Training uses weighted cross-entropy (for class imbalance) and MSE loss with a combined objective:
```
L = α · L_intent + (1-α) · L_complexity
```

### 4. Hybrid Routing with LightGBM

The router uses 27 features including:
- Text features (word count, character count, average word length)
- Keyword features (math/code/reasoning keyword counts)
- MTL outputs (intent probabilities, complexity score, confidence)
- Structural features (has_code_fence, sentence_count, punctuation_density)

**Strong-Inheritance**: If any segment in a dependency chain requires a strong model, all connected segments are upgraded to maintain reasoning integrity.

### 5. Speculative Execution

When a parent segment's confidence is very high (>0.95), child segments are speculatively executed in parallel:
- If the parent's actual output matches the speculated context → use cached result
- If mismatch → re-execute child with correct context
- Tracks hit/miss rates for adaptive threshold tuning

### 6. Context Blackboard

A shared memory structure that passes distilled context between dependent segments:
- **Key-Value Extraction**: `answer = 42` → `{"key": "answer", "value": "42"}`
- **Code Signature Detection**: Function definitions and class names
- **Numeric Result Extraction**: All numbers with context
- **Token Savings**: Typically reduces context tokens by 60-80% vs raw passing

### 7. LLM-as-a-Judge

For weak-model outputs on independent segments:
1. Run a 50-token self-check prompt: *"Rate this response 1-5 for accuracy"*
2. If score ≤ 2 → escalate to strong model
3. Track escalation rates to calibrate routing thresholds

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- ~4GB disk space for ML models

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd final_project

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python3 -m spacy download en_core_web_sm

# (Optional) Install benepar for constituency parsing
pip install benepar
python3 -c "import benepar; benepar.download('benepar_en3')"
```

### Dependencies (key packages)

```
torch>=2.0
transformers>=4.30
sentence-transformers>=2.2
lightgbm>=4.0
spacy>=3.5
numpy>=1.24
scikit-learn>=1.3
tenacity>=8.0
```

---

## Quick Start

### 1. Generate Dataset
```bash
cd dataset
python3 synthetic_dataset.py --num-samples 10000 --output dataset.json --seed 42
```

### 2. Train Intent/Complexity MTL Model
```bash
cd Intent_complexity
python3 intent_trainer.py \
    --data ../dataset/dataset.json \
    --output ../models/intent_complexity_mtl \
    --epochs 15 \
    --batch-size 32 \
    --lr 2e-5
```

### 3. Train Router Model
```bash
cd router
python3 train_router.py \
    --data ../dataset/dataset.json \
    --intent-model ../models/intent_complexity_mtl \
    --output ../models/router_model.txt \
    --config-output ../models/router_config.json
```

### 4. Run Pipeline Evaluation
```bash
# Simulated mode (no API key needed)
python3 pipeline.py \
    --data dataset/dataset.json \
    --backend simulated \
    --max-samples 500 \
    --output evaluation/pipeline_results.json \
    --save-outputs

# With Groq API (real LLM inference)
export GROQ_API_KEY="your-key-here"
python3 pipeline.py \
    --data dataset/dataset.json \
    --backend groq \
    --mode comparison \
    --max-samples 50 \
    --output evaluation/pipeline_results.json \
    --save-outputs
```

### 5. Interactive Mode
```bash
python3 run_interactive.py
```

---

## Detailed Usage

### Dataset Generation

The synthetic dataset generator creates prompts with:
- **11 intent categories** spanning strong-model (math, code, simulation, research, prediction, data_analysis) and weak-model (translation, summarization, explanation, communication, documentation) tasks
- **Multi-task prompts**: 60% of prompts contain 2+ segments with dependencies
- **40% single-task prompts** for baseline routing evaluation
- **Realistic noise**: Typos, informal language, mixed formatting
- **Ground truth labels**: Intent, complexity score, model requirement, dependencies

```bash
# Basic generation
python3 dataset/synthetic_dataset.py --num-samples 10000

# With LLM paraphrasing (requires Gemini API key)
export GEMINI_API_KEY="your-key"
python3 dataset/synthetic_dataset.py --num-samples 10000 --use-llm

# Custom ratios
python3 dataset/synthetic_dataset.py \
    --num-samples 10000 \
    --multi-task-ratio 0.60 \
    --dependent-ratio 0.40
```

### Model Training

#### Intent + Complexity MTL Trainer

```bash
python3 Intent_complexity/intent_trainer.py \
    --data dataset/dataset.json \
    --output models/intent_complexity_mtl \
    --epochs 15 \
    --batch-size 32 \
    --lr 2e-5 \
    --weight-decay 0.01 \
    --warmup-ratio 0.1
```

**Expected output:**
- Intent accuracy: ~80% (weighted F1: ~83%)
- Complexity MSE: ~0.008 (correlation: ~0.90)

#### Router Trainer

```bash
python3 router/train_router.py \
    --data dataset/dataset.json \
    --intent-model models/intent_complexity_mtl \
    --output models/router_model.txt \
    --config-output models/router_config.json
```

**Expected output:**
- Routing accuracy: ~86% (F1: ~88%)
- Optimal threshold: ~0.45
- Evaluation plots saved to `evaluation/`

### Pipeline Execution

```bash
# Full pipeline evaluation (simulated, no API key)
python3 pipeline.py \
    --data dataset/dataset.json \
    --backend simulated \
    --max-samples 500 \
    --output evaluation/pipeline_results.json \
    --save-outputs \
    --mode pipeline

# Single prompt test
python3 pipeline.py \
    --prompt "Solve the integral of x^2 * sin(x) dx and then explain the result"

# With real LLM backends (requires API key)
python3 pipeline.py \
    --backend groq \
    --mode comparison \
    --prompt "Write a Python quicksort and explain its time complexity"

# Skip safety ML models for faster startup
python3 pipeline.py \
    --skip-safety-ml \
    --prompt "Translate hello to French"
```

---

## Interactive Mode

```bash
python3 run_interactive.py
```

The interactive runner provides:
1. **API Key Configuration**: Set Groq, Gemini, OpenAI, HuggingFace tokens
2. **Dataset Evaluation**: Run pipeline on any dataset
3. **Interactive Prompt Testing**: Enter your own prompts
4. **Comparison Mode**: Pipeline vs strong-model baseline (needs API key)
5. **Results Viewer**: Display last evaluation results

API keys are persisted in `.env` file.

---

## Configuration

### Labels (`config/labels.py`)

```python
INTENT_LABELS = [
    "math", "code", "simulation", "research", "prediction",
    "data_analysis", "translation", "summarization", "explanation",
    "communication", "documentation"
]

STRONG_INTENTS = {"math", "code", "simulation", "research", "prediction", "data_analysis"}
WEAK_INTENTS = {"translation", "summarization", "explanation", "communication", "documentation"}
```

### Environment Variables (`.env`)

```env
GROQ_API_KEY="gsk_..."
GEMINI_API_KEY="AIza..."
OPENAI_API_KEY="sk-..."
HF_TOKEN="hf_..."
```

---

## Results

### Current Dataset (10,000 prompts, Original Synthetic)

| Metric | Value |
|--------|-------|
| **MTL Intent Accuracy** | 80.11% |
| **MTL Intent F1 (weighted)** | 82.79% |
| **MTL Complexity MSE** | 0.0084 |
| **MTL Complexity Correlation** | 0.9013 |
| **Router Accuracy** | 85.99% |
| **Router F1** | 87.89% |
| **Router Optimal Threshold** | 0.45 |

### Pipeline Evaluation (500 prompts, Simulated)

| Metric | Value |
|--------|-------|
| **Mean Latency** | 130.99 ms/prompt |
| **Median Latency** | 132.30 ms/prompt |
| **P95 Latency** | 179.74 ms/prompt |
| **Multi-task Rate** | 79.2% |
| **Cost Savings** | 49.1% routed to weak model |
| **Intent Accuracy (vs GT)** | 62.04% |
| **Routing Accuracy (vs GT)** | 58.57% |
| **Strong-Inheritance Upgrades** | 71 |
| **Parallelism Ratio** | 0.508 |
| **Safety Blocked** | 25 segments |

### Timing Breakdown (per prompt)

| Stage | Time (ms) | % |
|-------|-----------|---|
| Safety Pre-check | 44.3 | 33.9% |
| Safety Post-check | 28.5 | 21.8% |
| Verification | 24.4 | 18.6% |
| Execution | 15.7 | 12.0% |
| Decomposition | 9.1 | 7.0% |
| Classification | 8.4 | 6.4% |
| Routing | 0.5 | 0.4% |
| **Total** | **130.9** | **100%** |

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: tenacity` | `pip install tenacity` |
| `SDPA attention error` | Already handled — falls back to eager attention |
| `ModelRegistry fails to load` | Ensure `models/` directory has trained checkpoints |
| `spacy model not found` | `python3 -m spacy download en_core_web_sm` |
| `CUDA out of memory` | Add `--skip-safety-ml` flag or use CPU |
| `GROQ_API_KEY not set` | Use `--backend simulated` or set key in `.env` |

### GPU Memory Requirements

| Configuration | VRAM Required |
|--------------|---------------|
| Full pipeline (all models) | ~6 GB |
| Skip safety ML models | ~3 GB |
| CPU only | 0 (slower, ~3-5x) |

---

## What's New in v2

| Feature | v1 | v2 |
|---------|----|----|
| **Multi-task ratio** | 90% | **60%** (more balanced) |
| **3-segment chains** | ✗ | **✓** (depth-3 DAGs) |
| **Dependent pairs** | 10 | **20** (diverse DAGs) |
| **Intent accuracy** | 80.11% | **83.32%** (+3.2%) |
| **Intent F1 (weighted)** | 82.79% | **85.32%** (+2.5%) |
| **Complexity MSE** | 0.0084 | **0.0073** (-13%) |
| **Complexity ρ** | 0.9013 | **0.9084** |
| **Router F1** | 87.89% | **88.04%** |
| **Intent categories** | 11 (skewed) | **11 (balanced 7-14%)** |

---

## Quick Start

```bash
# 1. Generate enhanced dataset
PYTHONPATH=. python3 dataset/synthetic_dataset.py \
    --num-samples 10000 \
    --output dataset/dataset_v2.json \
    --multi-task-ratio 0.60 \
    --dependent-ratio 0.50 \
    --triple-ratio 0.15

# 2. Train MTL model
PYTHONPATH=. python3 Intent_complexity/intent_trainer.py \
    --data dataset/dataset_v2.json \
    --output models/intent_complexity_mtl_v2 \
    --epochs 15 --batch-size 32

# 3. Train Router
PYTHONPATH=. python3 router/train_router.py \
    --data dataset/dataset_v2.json \
    --output-dir models \
    --eval-dir evaluation

# 4. Run Pipeline
PYTHONPATH=. python3 pipeline.py \
    --data dataset/dataset_v2.json \
    --backend simulated \

    --max-samples 500 \
    --output evaluation/pipeline_results_v2.json \
    --save-outputs

# 5. Interactive mode
python3 run_interactive.py
```

---

## Architecture

The pipeline follows 8 stages:

```
User Prompt
    │
    ├─► [1] Safety Pre-Check (toxicity + injection + PII)
    ├─► [2] Semantic Decomposer + DAG Builder + DependencyAnalyzer
    ├─► [3] Intent/Complexity MTL (DistilBERT dual-head)
    ├─► [4] Hybrid Router + Strong-Inheritance Propagation
    ├─► [5] DAG Execution (parallel + speculative + blackboard)
    ├─► [6] LLM-as-a-Judge Quality Escalation
    ├─► [7] Safety Post-Check (output scanning)
    └─► [8] NLI Verification + Intelligent Aggregation
              │
              ▼
        Final Answer
```

All 8 ML models are eagerly loaded via a singleton **ModelRegistry** (~22s startup).

---

## Directory Structure

```
final_project/
├── pipeline.py                     # 8-stage orchestrator
├── model_registry.py               # Eager model loader
├── run_interactive.py              # Interactive runner + API keys
├── .env                            # API key storage
├── config/labels.py                # 11 intent labels
├── dataset/
│   ├── synthetic_dataset.py        # Generator (60/40, 3-seg chains)
│   ├── dataset.json                # Original dataset (v1)
│   └── dataset_v2.json             # Enhanced dataset (v2)
├── Intent_complexity/
│   ├── intent_estimator.py         # MTL inference
│   └── intent_trainer.py           # MTL training
├── router/
│   ├── router.py                   # LightGBM + strong-inheritance
│   └── train_router.py             # Router training
├── semantic_decom_dependency/
│   └── decomposition_dependency.py # Decomposer + DAG + DependencyAnalyzer
├── execution_engine/
│   ├── execution_engine.py         # DAG executor + speculative
│   └── blackboard.py              # Context blackboard + distiller
├── verification_aggregation/
│   ├── aggregator.py              # NLI verification + merging
│   └── quality_judge.py           # LLM-as-a-Judge
├── safety_system/
│   └── safety_system.py           # 3-layer safety
├── models/                        # Trained checkpoints
├── evaluation/                    # Results + plots
├── report_v1.tex                  # ACL report (v1 dataset)
├── report_v2.tex                  # ACL report (v2 dataset)
└── README.md / README_v2.md
```

---

## v2 Results Summary

### MTL Model (DistilBERT, 15 epochs)

| Metric | Score |
|--------|-------|
| Intent Accuracy | **83.32%** |
| Intent F1 (weighted) | **85.32%** |
| Intent F1 (macro) | **86.73%** |
| Complexity MSE | **0.0073** |
| Complexity MAE | **0.0552** |
| Complexity Correlation | **0.9084** |

### Router (LightGBM)

| Metric | Score |
|--------|-------|
| Test Accuracy | **85.66%** |
| Test F1 | **88.04%** |
| AUC-ROC | **85.37%** |
| Optimal Threshold | **0.50** |
| Top Feature | complexity_score (gain=0.986) |

### Pipeline (500 prompts, simulated)

| Metric | Value |
|--------|-------|
| Mean Latency | 131 ms/prompt |
| Cost Savings | 49.1% weak-routed |
| Strong-Inheritance Upgrades | 71 |
| Parallelism Ratio | 0.508 |
| Safety Blocked | 25 segments |

---

## API Key Configuration

Edit `.env` or use the interactive runner:

```env
GROQ_API_KEY="gsk_..."
GEMINI_API_KEY="AIza..."
OPENAI_API_KEY="sk-..."
HF_TOKEN="hf_..."
```

---

## License

Academic use — INLP Final Project.


This project is developed for academic purposes as part of the INLP (Introduction to NLP) course final project.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{asp2026,
  title={Adaptive Semantic Decomposition and Parallelism for Cost-Efficient LLM Routing},
  author={Team Nier_INLP Vidvathama R , Sarvesh Sachin, Snehil Sanjog , Chaitanya D},
  year={2026},
  note={INLP Final Project}
}
```
