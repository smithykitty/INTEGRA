"""
config.py — Central configuration for the Integrative Medicine HITL-RL system.

Tuned for Open OnDemand / UVA Rivanna JupyterLab.

Set your API key before running:
    export ANTHROPIC_API_KEY=sk-ant-...
Or paste it directly into ANTHROPIC_API_KEY below (don't commit to GitHub).
"""

import os
from pathlib import Path

# ── All data lives in your home directory — persistent across OOD sessions
ROOT = Path.home() / "integrative_rl"

DATA_DIR     = ROOT / "data"
CHROMA_DIR   = DATA_DIR / "chroma_db"
EPISODES_DIR = DATA_DIR / "episodes"
FIGURES_DIR  = DATA_DIR / "figures"
MODELS_DIR   = ROOT / "models"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"

for d in [DATA_DIR, CHROMA_DIR, EPISODES_DIR, FIGURES_DIR, MODELS_DIR, RAW_DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Database
DB_PATH = EPISODES_DIR / "rl_episodes.db"

# ── API key — reads environment variable first
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Models
LLM_MODEL   = "claude-sonnet-4-6"
EMBED_MODEL = "all-MiniLM-L6-v2"

# ── Tradition → ChromaDB collection name mapping
TRADITION_COLS = {
    "TCM":                 "tcm",
    "Ayurveda":            "ayurveda",
    "Naturopathy":         "naturopathy",
    "Clinical Herbalism":  "clinical_herbalism",
    "Functional Nutrition":"functional_nutrition",
}

# ── RL reward shaping weights
REWARD_CONFIG = {
    "approve":              1.0,
    "modify":               0.3,
    "reject":              -0.5,
    "cross_synergy_bonus":  0.25,
    "evidence_A_bonus":     0.20,
    "evidence_C_penalty":  -0.10,
    "herb_drug_penalty":   -0.80,
    "high_priority_weight": 1.30,
}

# ── Feature encoding lists
SYSTEMS    = ["TCM","Ayurveda","Naturopathy","Clinical Herbalism","Functional Nutrition","Integrated"]
CATEGORIES = ["Herbal formula","Acupuncture/bodywork","Diet & nutrition","Targeted supplementation",
              "Lifestyle medicine","Mind-body practice","Detox & elimination",
              "Functional testing","Adaptogenic protocol","Gut restoration"]
PRIORITIES = ["High","Medium","Low"]
EVIDENCE   = ["A – RCT evidence","B – observational / traditional clinical","C – traditional use only"]

if __name__ == "__main__":
    print("Project root:", ROOT)
    print("Data dir:    ", DATA_DIR)
    print("DB path:     ", DB_PATH)
    print("API key set: ", bool(ANTHROPIC_API_KEY))
