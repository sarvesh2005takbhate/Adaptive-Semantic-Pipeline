"""
Canonical label definitions used across ALL modules.
Import this everywhere instead of hardcoding labels.
"""

# ── Intent Labels (11 classes) ──────────────────────────────
INTENT_LABELS = [
    "math",
    "code",
    "simulation",
    "research",
    "prediction",
    "data_analysis",
    "translation",
    "summarization",
    "explanation",
    "communication",
    "documentation",
]

LABEL2ID = {label: i for i, label in enumerate(INTENT_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(INTENT_LABELS)}
NUM_INTENTS = len(INTENT_LABELS)

# ── Routing Tiers ───────────────────────────────────────────
STRONG_INTENTS = {"math", "code", "simulation", "research", "prediction", "data_analysis"}
WEAK_INTENTS   = {"translation", "summarization", "explanation", "communication", "documentation"}

# ── Router Targets ──────────────────────────────────────────
ROUTE_WEAK   = "weak_model"
ROUTE_STRONG = "strong_model"
ROUTE_BLOCK  = "safe_block"
ROUTE_VERIFY = "verify_required"