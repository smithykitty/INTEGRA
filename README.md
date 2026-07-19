# Integrative Medicine HITL-RL — Google Colab Package

Five-notebook reinforcement learning system for five-tradition integrative medicine care plan generation with clinician-in-the-loop feedback.

## Traditions covered
- Traditional Chinese Medicine (TCM)
- Ayurveda
- Naturopathic medicine
- Clinical herbalism (Western Eclectic, evidence-graded)
- Functional nutrition & wellness

## Notebooks

| Notebook | Purpose | Runtime |
|---|---|---|
| NB1_knowledge_ingestion | Embed tradition texts into ChromaDB vector store | CPU |
| NB2_patient_intake_rag | Patient intake form + RAG plan generation via Claude API | CPU |
| NB3_clinician_review | Per-item approve/modify/reject + reward computation + SQLite store | CPU |
| NB4_rl_training | Offline RL training — reward predictor policy | T4 GPU |
| NB5_analytics_dashboard | Live Plotly dashboard — reward curves, heatmaps, approval stats | CPU |

## Setup

1. Upload all notebooks to Google Colab
2. In Colab: `Secrets` → add `ANTHROPIC_API_KEY`
3. Run NB1 first (once) to build the vector store
4. Run NB2 to generate a care plan → note the Episode ID
5. Run NB3 to review the episode
6. After 10+ episodes, run NB4 to train the RL policy
7. Run NB5 anytime to view analytics

## Data schema

```
Google Drive/integrative_rl/
  chroma_db/          ← ChromaDB vector store (5 tradition collections)
  raw_docs/           ← Place PDFs here for ingestion in NB1
  episodes/
    {episode_id}.json ← Draft episodes from NB2
    rl_episodes.db    ← Reviewed episodes (SQLite, NB3 → NB4)
  rl_policy/
    reward_predictor.pt ← Trained policy weights
    training_summary.png
```

## Reward design

| Action | Base reward |
|---|---|
| Approve | +1.0 |
| Modify | +0.3 |
| Reject | -0.5 |
| Cross-system synergy bonus | +0.25 |
| Grade A evidence bonus | +0.2 |
| Grade C evidence penalty | -0.1 |
| Herb-drug safety violation | -0.8 |
| High-priority item multiplier | ×1.3 |

Adjust in `NB3 → REWARD_CONFIG`.

## Extending the system

- **Add PDFs**: drop into `raw_docs/` and uncomment the PDF ingestion block in NB1
- **Add traditions**: extend `TRADITIONS` dict in NB1, add collection name to NB2 `TRADITION_COLS`
- **Upgrade RL**: replace the reward predictor in NB4 with `d3rlpy` CQL for true offline RL
- **Multi-reviewer**: pass different `reviewer_id` values in NB3; compare inter-rater agreement in NB5
