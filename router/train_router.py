"""
Router Model Trainer — Adaptive Semantic Parallelism
=====================================================
Trains a LightGBM binary classifier to predict: strong_model (1) vs weak_model (0).

Research-grade features:
  - Stratified train/val/test (70/15/15)
  - Feature importance analysis (gain, split, SHAP-like permutation)
  - Threshold sensitivity analysis
  - Calibration curve for P(strong)
  - Per-intent routing accuracy
  - Decision boundary visualization
  - Ablation: heuristic-only vs ML vs hybrid comparison
  - Cross-validation with confidence intervals

Usage:
    python router_trainer.py --data data/dataset.jsonl
    python router_trainer.py --data data/dataset.jsonl --cross-val 5

CRITICAL NOTE ON LABELS:
  model_requirement is assigned using stochastic + complexity-aware logic
  to break deterministic mapping from intent → route.

  This enables the router to learn true difficulty-based routing instead of
  memorizing intent patterns.

  Future work:
    1. Evaluate strong vs weak model outputs
    2. Use quality metrics (BERTScore / human eval)
    3. Label true routing need
    4. Retrain on empirical labels

  This trainer works with both deterministic and empirical labels.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.labels import (
    INTENT_LABELS,
    LABEL2ID,
    NUM_INTENTS,
    ROUTE_STRONG,
    ROUTE_WEAK,
    STRONG_INTENTS,
    WEAK_INTENTS,
)
from router import (
    FEATURE_NAMES,
    NUM_FEATURES,
    extract_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger("router_trainer")


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_router_data(data_path: str) -> pd.DataFrame:
    """Load dataset and extract segments with routing labels."""
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            for seg in item.get("segments", []):
                text = seg.get("text", "").strip()
                intent = seg.get("intent", "")
                model_req = seg.get("model_requirement", "")
                complexity = seg.get("complexity_score", 0.5)

                if not text or intent not in LABEL2ID:
                    continue
                if model_req not in ("strong_model", "weak_model"):
                    continue

                records.append({
                    "text": text,
                    "intent": intent,
                    "complexity_score": complexity,
                    "model_requirement": model_req,
                    "label": 1 if model_req == "strong_model" else 0,
                    "depends_on": seg.get("depends_on", []),
                })

    df = pd.DataFrame(records)
    log.info(f"Loaded {len(df)} segments from {data_path}")
    log.info(f"Label distribution: strong={sum(df['label']==1)}, weak={sum(df['label']==0)}")
    log.info(f"Intent distribution:\n{df['intent'].value_counts().to_string()}")

    return df


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build the feature matrix from dataframe rows."""
    features = []
    for _, row in df.iterrows():
        seg = {
            "text": row["text"],
            "intent": row["intent"],
            "complexity_score": row["complexity_score"],
            "intent_confidence": 0.50,  # placeholder until real pipeline
            "depth": 0,
            "depends_on": row.get("depends_on", []),
            "unsafe_candidate": False,
        }
        features.append(extract_features(seg))
    return np.array(features, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════

def train_router(
    df: pd.DataFrame,
    output_dir: str = "models",
    eval_dir: str = "evaluation",
    num_rounds: int = 200,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    early_stopping_rounds: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Full training pipeline with comprehensive evaluation."""

    output_path = Path(output_dir)
    eval_path = Path(eval_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    eval_path.mkdir(parents=True, exist_ok=True)

    # ── Build features ──────────────────────────────────────
    log.info("Extracting features...")
    X = build_feature_matrix(df)
    y = df["label"].values

    log.info(f"Feature matrix shape: {X.shape}")
    assert X.shape[1] == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {X.shape[1]}"

    # ── Stratified 70/15/15 split ───────────────────────────
    df["pattern"] = df["text"].apply(
    lambda x: " ".join(str(x).lower().split()[:3]) if isinstance(x, str) else ""
)

    unique_patterns = df["pattern"].dropna().unique().tolist()

    train_p, temp_p = train_test_split(unique_patterns, test_size=0.3, random_state=seed)
    temp_p = list(temp_p) 


    train_idx = df["pattern"].isin(train_p)
    temp_idx = df["pattern"].isin(temp_p)

    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]
    idx_temp = np.where(temp_idx)[0]

    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp,
        test_size=0.50, random_state=seed, stratify=y_temp,
    )

    log.info(f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
        # sanity: same intent → multiple routes
    intent_route_map = df.groupby("intent")["label"].nunique()
    log.info(f"Intent → unique routes:\n{intent_route_map}")

    # ── Train LightGBM ──────────────────────────────────────
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_data)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 10,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": seed,
    }

    log.info("Training LightGBM...")
    start_time = time.time()

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_rounds,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    training_time = time.time() - start_time
    log.info(f"Training completed in {training_time:.1f}s ({model.best_iteration} rounds)")

    # ── Save model ──────────────────────────────────────────
    model_path = output_path / "router_model.txt"
    model.save_model(str(model_path))
    log.info(f"Model saved → {model_path}")

    # ── Evaluate all splits ─────────────────────────────────
    val_results = evaluate_split(model, X_val, y_val, df.iloc[idx_val], "validation", eval_path)
    test_results = evaluate_split(model, X_test, y_test, df.iloc[idx_test], "test", eval_path)

    # ── Feature importance ──────────────────────────────────
    importance_results = analyze_feature_importance(model, X_test, y_test, eval_path)

    # ── Threshold sensitivity ───────────────────────────────
    threshold_results = analyze_thresholds(model, X_test, y_test, eval_path)

    # ── Per-intent routing accuracy ─────────────────────────
    per_intent = analyze_per_intent(model, X_test, y_test, df.iloc[idx_test], eval_path)

    # ── Ablation: heuristic vs ML ───────────────────────────
    ablation = ablation_study(model, X_test, y_test, df.iloc[idx_test])

    # ── Calibration ─────────────────────────────────────────
    calibration = analyze_calibration(model, X_test, y_test, eval_path)

    # ── Compile results ─────────────────────────────────────
    results = {
        "model": "LightGBM",
        "num_features": NUM_FEATURES,
        "feature_names": FEATURE_NAMES,
        "training": {
            "num_rounds": model.best_iteration,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "training_time_seconds": round(training_time, 1),
            "train_size": len(y_train),
            "val_size": len(y_val),
            "test_size": len(y_test),
        },
        "validation": val_results,
        "test": test_results,
        "feature_importance": importance_results,
        "threshold_analysis": threshold_results,
        "per_intent_accuracy": per_intent,
        "ablation": ablation,
        "calibration": calibration,
        "label_distribution": {
            "strong": int(sum(y == 1)),
            "weak": int(sum(y == 0)),
            "strong_pct": round(sum(y == 1) / len(y) * 100, 1),
        },
    }

    # Save config
    config = {
        "feature_names": FEATURE_NAMES,
        "num_features": NUM_FEATURES,
        "threshold_strong": 0.65,
        "threshold_weak": 0.35,
        "intent_labels": INTENT_LABELS,
    }
    with open(output_path / "router_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save full results
    results_path = eval_path / "router_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Results saved → {results_path}")

    return results


# ═══════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_split(
    model: lgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
    df_slice: pd.DataFrame,
    split_name: str,
    eval_path: Path,
) -> Dict[str, Any]:
    """Comprehensive evaluation on a data split."""
    probs = model.predict(X)
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="binary")
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    ll = log_loss(y, probs)

    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = 0.0

    # Confusion matrix
    cm = confusion_matrix(y, preds)
    _plot_router_confusion_matrix(cm, eval_path / f"router_cm_{split_name}.png")

    # ROC curve
    _plot_roc_curve(y, probs, eval_path / f"router_roc_{split_name}.png")

    # Precision-recall curve
    _plot_pr_curve(y, probs, eval_path / f"router_pr_{split_name}.png")

    log.info(f"\n{'='*50}")
    log.info(f"  ROUTER {split_name.upper()} EVALUATION")
    log.info(f"{'='*50}")
    log.info(f"  Accuracy:   {acc:.4f}")
    log.info(f"  F1:         {f1:.4f}")
    log.info(f"  Precision:  {prec:.4f}")
    log.info(f"  Recall:     {rec:.4f}")
    log.info(f"  AUC-ROC:    {auc:.4f}")
    log.info(f"  Log-loss:   {ll:.4f}")
    log.info(f"  Confusion:\n{cm}")

    return {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "auc_roc": round(auc, 4),
        "log_loss": round(ll, 4),
        "confusion_matrix": cm.tolist(),
    }


# ═══════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════

def analyze_feature_importance(
    model: lgb.Booster,
    X_test: np.ndarray,
    y_test: np.ndarray,
    eval_path: Path,
) -> Dict[str, Any]:
    """Analyze feature importance using gain, split count, and permutation."""

    # LightGBM built-in importance
    gain_imp = model.feature_importance(importance_type="gain")
    split_imp = model.feature_importance(importance_type="split")

    # Normalize
    gain_norm = gain_imp / (gain_imp.sum() + 1e-8)
    split_norm = split_imp / (split_imp.sum() + 1e-8)

    # Permutation importance (more reliable)
    baseline_preds = (model.predict(X_test) > 0.5).astype(int)
    baseline_acc = accuracy_score(y_test, baseline_preds)
    perm_importance = []

    for i in range(NUM_FEATURES):
        X_perm = X_test.copy()
        np.random.shuffle(X_perm[:, i])
        perm_preds = (model.predict(X_perm) > 0.5).astype(int)
        perm_acc = accuracy_score(y_test, perm_preds)
        perm_importance.append(round(baseline_acc - perm_acc, 6))

    # Plot top features
    _plot_feature_importance(FEATURE_NAMES, gain_norm, perm_importance, eval_path / "router_feature_importance.png")

    # Build result
    feature_data = {}
    for i, name in enumerate(FEATURE_NAMES):
        feature_data[name] = {
            "gain": round(float(gain_norm[i]), 4),
            "split_count": int(split_imp[i]),
            "permutation_drop": round(perm_importance[i], 4),
        }

    # Sort by gain
    top_features = sorted(feature_data.items(), key=lambda x: x[1]["gain"], reverse=True)

    log.info("\n  TOP 10 FEATURES BY GAIN:")
    for name, vals in top_features[:10]:
        log.info(f"    {name:25s}  gain={vals['gain']:.4f}  perm_drop={vals['permutation_drop']:.4f}")

    return {
        "by_feature": feature_data,
        "top_10_gain": [name for name, _ in top_features[:10]],
    }


def _plot_feature_importance(names, gain, perm, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by gain
    idx = np.argsort(gain)[::-1][:15]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Gain
    axes[0].barh(range(len(idx)), [gain[i] for i in idx], color="steelblue")
    axes[0].set_yticks(range(len(idx)))
    axes[0].set_yticklabels([names[i] for i in idx])
    axes[0].set_title("Feature Importance (Gain)")
    axes[0].invert_yaxis()

    # Permutation
    perm_idx = np.argsort(perm)[::-1][:15]
    axes[1].barh(range(len(perm_idx)), [perm[i] for i in perm_idx], color="coral")
    axes[1].set_yticks(range(len(perm_idx)))
    axes[1].set_yticklabels([names[i] for i in perm_idx])
    axes[1].set_title("Permutation Importance (Accuracy Drop)")
    axes[1].invert_yaxis()

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved feature importance → {path}")


# ═══════════════════════════════════════════════════════════════
#  THRESHOLD ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_thresholds(
    model: lgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
    eval_path: Path,
) -> Dict[str, Any]:
    """Analyze routing accuracy at different threshold settings."""
    probs = model.predict(X)

    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []

    for t in thresholds:
        preds = (probs > t).astype(int)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, zero_division=0)
        strong_pct = preds.mean() * 100
        results.append({
            "threshold": round(float(t), 2),
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "strong_routed_pct": round(strong_pct, 1),
        })

    # Plot
    _plot_threshold_analysis(results, eval_path / "router_threshold_analysis.png")

    # Find optimal
    best = max(results, key=lambda x: x["f1"])
    log.info(f"  Optimal threshold: {best['threshold']} (F1={best['f1']}, Acc={best['accuracy']})")

    return {
        "thresholds": results,
        "optimal_threshold": best["threshold"],
        "optimal_f1": best["f1"],
    }


def _plot_threshold_analysis(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = [r["threshold"] for r in results]
    accs = [r["accuracy"] for r in results]
    f1s = [r["f1"] for r in results]
    strongs = [r["strong_routed_pct"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(ts, accs, "b-o", label="Accuracy", markersize=3)
    ax1.plot(ts, f1s, "r-s", label="F1", markersize=3)
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Score")
    ax1.legend(loc="center left")

    ax2 = ax1.twinx()
    ax2.plot(ts, strongs, "g--", label="% Routed Strong", alpha=0.6)
    ax2.set_ylabel("% Strong Routed")
    ax2.legend(loc="center right")

    ax1.set_title("Threshold Sensitivity Analysis")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  PER-INTENT ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_per_intent(
    model: lgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
    df_slice: pd.DataFrame,
    eval_path: Path,
) -> Dict[str, Any]:
    """Routing accuracy broken down by intent."""
    probs = model.predict(X)
    preds = (probs > 0.5).astype(int)
    intents = df_slice["intent"].values

    per_intent = {}
    for intent in INTENT_LABELS:
        mask = intents == intent
        if mask.sum() == 0:
            continue
        intent_acc = accuracy_score(y[mask], preds[mask])
        intent_probs = probs[mask]
        per_intent[intent] = {
            "n_samples": int(mask.sum()),
            "accuracy": round(float(intent_acc), 4),
            "mean_prob_strong": round(float(intent_probs.mean()), 4),
            "std_prob_strong": round(float(intent_probs.std()), 4),
            "expected_route": "strong" if intent in STRONG_INTENTS else "weak",
        }

    # Plot
    _plot_per_intent(per_intent, eval_path / "router_per_intent.png")

    return per_intent


def _plot_per_intent(per_intent, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(per_intent.keys())
    probs = [per_intent[n]["mean_prob_strong"] for n in names]
    colors = ["coral" if per_intent[n]["expected_route"] == "strong" else "steelblue" for n in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, probs, color=colors, edgecolor="black", alpha=0.8)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=0.65, color="red", linestyle="--", alpha=0.3, label="Strong threshold (0.65)")
    ax.axhline(y=0.35, color="blue", linestyle="--", alpha=0.3, label="Weak threshold (0.35)")
    ax.set_ylabel("Mean P(strong)")
    ax.set_title("Router Confidence by Intent (red=strong expected, blue=weak expected)")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  ABLATION STUDY
# ═══════════════════════════════════════════════════════════════

def ablation_study(
    model: lgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
    df_slice: pd.DataFrame,
) -> Dict[str, Any]:
    """Compare: heuristic-only vs ML-only vs hybrid routing."""

    # ML-only (threshold=0.5)
    ml_preds = (model.predict(X) > 0.5).astype(int)
    ml_acc = accuracy_score(y, ml_preds)

    # Heuristic-only
    heuristic_preds = []
    for _, row in df_slice.iterrows():
        intent = row["intent"]
        if intent in STRONG_INTENTS:
            heuristic_preds.append(1)
        else:
            if row["complexity_score"] > 0.6:
                heuristic_preds.append(1)
            else:
                heuristic_preds.append(0)
    heuristic_preds = np.array(heuristic_preds)
    heuristic_acc = accuracy_score(y, heuristic_preds)

    # Hybrid (ML with heuristic fallback in uncertain zone)
    probs = model.predict(X)
    hybrid_preds = []
    for i, prob in enumerate(probs):
        if prob > 0.65:
            hybrid_preds.append(1)
        elif prob < 0.35:
            hybrid_preds.append(0)
        else:
            # Fallback to heuristic
            hybrid_preds.append(heuristic_preds[i])
    hybrid_preds = np.array(hybrid_preds)
    hybrid_acc = accuracy_score(y, hybrid_preds)

    uncertain_pct = ((probs >= 0.35) & (probs <= 0.65)).mean() * 100

    log.info("\n  ABLATION: Routing Strategy Comparison")
    log.info(f"    Heuristic-only: {heuristic_acc:.4f}")
    log.info(f"    ML-only:        {ml_acc:.4f}")
    log.info(f"    Hybrid:         {hybrid_acc:.4f}")
    log.info(f"    Uncertain zone: {uncertain_pct:.1f}%")

    return {
        "heuristic_accuracy": round(heuristic_acc, 4),
        "ml_accuracy": round(ml_acc, 4),
        "hybrid_accuracy": round(hybrid_acc, 4),
        "uncertain_zone_pct": round(uncertain_pct, 1),
    }


# ═══════════════════════════════════════════════════════════════
#  CALIBRATION
# ═══════════════════════════════════════════════════════════════

def analyze_calibration(
    model: lgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
    eval_path: Path,
) -> Dict[str, float]:
    """Calibration analysis for router probabilities."""
    probs = model.predict(X)

    try:
        fraction_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="uniform")
    except Exception:
        return {"ece": 0.0}

    # ECE
    bin_counts = np.histogram(probs, bins=10, range=(0, 1))[0]
    total = len(probs)
    ece = 0.0
    for i in range(len(fraction_pos)):
        if i < len(bin_counts):
            ece += (bin_counts[i] / total) * abs(fraction_pos[i] - mean_pred[i])

    # Plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, fraction_pos, "o-", color="steelblue", label="Router")
    ax.set_xlabel("Mean predicted P(strong)")
    ax.set_ylabel("Fraction actually strong")
    ax.set_title(f"Router Calibration (ECE={ece:.4f})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(eval_path / "router_calibration.png", dpi=150)
    plt.close(fig)

    return {"ece": round(ece, 4), "mean_prob": round(float(probs.mean()), 4)}


# ═══════════════════════════════════════════════════════════════
#  CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════

def cross_validate_router(
    df: pd.DataFrame,
    n_folds: int = 5,
    num_rounds: int = 200,
    learning_rate: float = 0.05,
    seed: int = 42,
    eval_dir: str = "evaluation/router_cv",
) -> Dict[str, Any]:
    """K-fold cross-validation for reporting mean ± std."""
    eval_path = Path(eval_dir)
    eval_path.mkdir(parents=True, exist_ok=True)

    X = build_feature_matrix(df)
    y = df["label"].values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        log.info(f"  Fold {fold_idx+1}/{n_folds}")
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES)
        val_data = lgb.Dataset(X_te, label=y_te, feature_name=FEATURE_NAMES, reference=train_data)

        params = {
            "objective": "binary", "metric": "binary_logloss",
            "num_leaves": 31, "learning_rate": learning_rate,
            "feature_fraction": 0.8, "verbose": -1, "seed": seed + fold_idx,
        }

        model = lgb.train(
            params, train_data, num_boost_round=num_rounds,
            valid_sets=[val_data], callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
        )

        probs = model.predict(X_te)
        preds = (probs > 0.5).astype(int)

        fold_results.append({
            "fold": fold_idx + 1,
            "accuracy": round(accuracy_score(y_te, preds), 4),
            "f1": round(f1_score(y_te, preds, zero_division=0), 4),
            "auc": round(roc_auc_score(y_te, probs) if len(set(y_te)) > 1 else 0.0, 4),
        })

    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]
    aucs = [r["auc"] for r in fold_results]

    summary = {
        "n_folds": n_folds,
        "folds": fold_results,
        "accuracy": f"{np.mean(accs):.4f} ± {np.std(accs):.4f}",
        "f1": f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
        "auc": f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
    }

    log.info(f"\n  CROSS-VALIDATION ({n_folds} folds)")
    log.info(f"    Accuracy: {summary['accuracy']}")
    log.info(f"    F1:       {summary['f1']}")
    log.info(f"    AUC:      {summary['auc']}")

    with open(eval_path / "router_cv_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ═══════════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════════

def _plot_router_confusion_matrix(cm, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Weak", "Strong"],
                yticklabels=["Weak", "Strong"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Router Confusion Matrix")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_roc_curve(y, probs, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fpr, tpr, _ = roc_curve(y, probs)
        auc = roc_auc_score(y, probs)
    except ValueError:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, "b-", label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Router ROC Curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_pr_curve(y, probs, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        prec, rec, _ = precision_recall_curve(y, probs)
    except ValueError:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(rec, prec, "b-")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Router Precision-Recall Curve")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train router model for Adaptive Semantic Parallelism")
    parser.add_argument("--data", type=str, default="dataset/dataset.json")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--eval-dir", type=str, default="evaluation")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--leaves", type=int, default=31)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cross-val", type=int, default=0)

    args = parser.parse_args()

    df = load_router_data(args.data)
    if len(df) == 0:
        log.error("No valid segments found!")
        return

    results = train_router(
        df,
        output_dir=args.output_dir,
        eval_dir=args.eval_dir,
        num_rounds=args.rounds,
        learning_rate=args.lr,
        num_leaves=args.leaves,
        seed=args.seed,
    )

    if args.cross_val > 1:
        cross_validate_router(df, n_folds=args.cross_val, seed=args.seed)

    # Print summary
    test = results["test"]
    abl = results["ablation"]
    print(f"\n{'='*60}")
    print(f"  ROUTER TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Test Accuracy:    {test['accuracy']}")
    print(f"  Test F1:          {test['f1']}")
    print(f"  Test AUC-ROC:     {test['auc_roc']}")
    print(f"  Heuristic-only:   {abl['heuristic_accuracy']}")
    print(f"  ML-only:          {abl['ml_accuracy']}")
    print(f"  Hybrid:           {abl['hybrid_accuracy']}")
    print(f"  Uncertain zone:   {abl['uncertain_zone_pct']}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()