"""
episode_store.py — SQLite persistence for HITL-RL episodes and reward computation.
"""

import json
import sqlite3
import datetime
from pathlib import Path
from config import DB_PATH, REWARD_CONFIG


# ── Database setup

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id     TEXT PRIMARY KEY,
            patient_json   TEXT,
            plan_json      TEXT,
            reviews_json   TEXT,
            reward_total   REAL,
            reward_breakdown TEXT,
            reviewed_at    TEXT,
            reviewer_id    TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS item_reviews (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id          TEXT,
            item_index          INTEGER,
            system              TEXT,
            category            TEXT,
            priority            TEXT,
            evidence_grade      TEXT,
            has_cross_synergy   INTEGER,
            has_herb_drug_flag  INTEGER,
            action              TEXT,
            clinician_note      TEXT,
            item_reward         REAL,
            reviewed_at         TEXT
        )
    """)
    con.commit()
    con.close()


# ── Reward computation

def compute_item_reward(action: str, item: dict, note: str = "") -> float:
    base = REWARD_CONFIG.get(action, 0.0)

    if item.get("priority") == "High":
        base *= REWARD_CONFIG["high_priority_weight"]

    if item.get("cross_system_synergy") and action == "approve":
        base += REWARD_CONFIG["cross_synergy_bonus"]

    grade = item.get("evidence_grade", "")
    if "A" in grade and action == "approve":
        base += REWARD_CONFIG["evidence_A_bonus"]
    elif "C" in grade and action == "approve":
        base += REWARD_CONFIG["evidence_C_penalty"]

    if item.get("herb_drug_flag") and action == "approve" and not note.strip():
        base += REWARD_CONFIG["herb_drug_penalty"]

    return round(base, 3)


# ── Save / load episodes

def save_draft_episode(episode_id: str, patient: dict, plan: list, episodes_dir: Path):
    """Save a draft episode JSON before clinician review."""
    path = episodes_dir / f"{episode_id}.json"
    with open(path, "w") as f:
        json.dump({"episode_id": episode_id, "patient": patient, "plan": plan}, f, indent=2)
    return path


def load_draft_episode(episode_id: str, episodes_dir: Path) -> dict:
    path = episodes_dir / f"{episode_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Episode {episode_id} not found at {path}")
    with open(path) as f:
        return json.load(f)


def save_reviewed_episode(
    episode_id: str,
    patient: dict,
    plan: list,
    reviews: dict,
    reviewer_id: str = "DR01",
):
    """Persist a fully reviewed episode to SQLite."""
    total_reward = sum(r["reward"] for r in reviews.values())
    reward_breakdown = {str(i): r["reward"] for i, r in reviews.items()}
    reviewed_at = datetime.datetime.utcnow().isoformat()

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute(
        """
        INSERT OR REPLACE INTO episodes
        (episode_id, patient_json, plan_json, reviews_json,
         reward_total, reward_breakdown, reviewed_at, reviewer_id)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            episode_id,
            json.dumps(patient),
            json.dumps(plan),
            json.dumps(reviews),
            total_reward,
            json.dumps(reward_breakdown),
            reviewed_at,
            reviewer_id,
        ),
    )

    for i, item in enumerate(plan):
        rev = reviews.get(i, {})
        cur.execute(
            """
            INSERT INTO item_reviews
            (episode_id, item_index, system, category, priority, evidence_grade,
             has_cross_synergy, has_herb_drug_flag, action, clinician_note,
             item_reward, reviewed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                episode_id, i,
                item.get("system"), item.get("category"),
                item.get("priority"), item.get("evidence_grade"),
                1 if item.get("cross_system_synergy") else 0,
                1 if item.get("herb_drug_flag") else 0,
                rev.get("action"), rev.get("note", ""),
                rev.get("reward"), reviewed_at,
            ),
        )

    con.commit()
    con.close()
    return total_reward


def load_all_episodes():
    """Load all reviewed episodes as DataFrames."""
    import pandas as pd
    con = sqlite3.connect(DB_PATH)
    items = pd.read_sql("SELECT * FROM item_reviews ORDER BY reviewed_at", con)
    eps   = pd.read_sql("SELECT * FROM episodes ORDER BY reviewed_at", con)
    con.close()
    return items, eps


def episode_count() -> int:
    try:
        con = sqlite3.connect(DB_PATH)
        n = con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        con.close()
        return n
    except Exception:
        return 0
