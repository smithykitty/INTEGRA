# Integrative Medicine HITL-RL — UVA Rivanna / Open OnDemand

Five-tradition integrative medicine care plan generator with clinician-in-the-loop
reinforcement learning. Runs in JupyterLab on UVA Rivanna via Open OnDemand.

---

## First-time setup

Open a **terminal** in JupyterLab (File → New → Terminal), then run each line:

```bash
# 1. Confirm Python version (must be 3.13)
python --version

# 2. Install all dependencies under the system Python
python -m pip install --user gradio anthropic chromadb sentence-transformers \
    torch scikit-learn pandas matplotlib seaborn plotly pypdf sqlalchemy tqdm

# 3. Verify gradio installed correctly
python -c "import gradio; print(gradio.__version__)"

# 4. Set your Anthropic API key permanently
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc && source ~/.bashrc

# 5. Build the ChromaDB knowledge vector store (one-time, ~30 seconds)
cd ~/integrative_rl
python src/knowledge_base.py
```

---

## Running the app (every session)

Open a terminal in JupyterLab and run:

```bash
cd ~/integrative_rl
python src/app.py
```

The app starts on port 7860 and prints a public Gradio link:
```
* Running on public URL: https://abc123def456.gradio.live
```

Click that link — it opens the app in any browser tab. The link stays live
as long as the terminal is running. If the terminal closes, just rerun
`python src/app.py` to get a new link.

---

## Important notes for Rivanna

**Python version:** Rivanna's JupyterLab runs Python 3.13. Always use `python`
(not `python3.11`) so packages install and load from the right location.

**No `module load` needed:** The system Python in JupyterLab terminals works
directly. Do not run `module load anaconda` or `module load python` — these
load different Python versions and cause package conflicts.

**Package location:** All packages install to `~/.local/lib/python3.13/site-packages/`.
If you ever see import errors after a fresh terminal, run:
```bash
export PYTHONPATH=$HOME/.local/lib/python3.13/site-packages:$PYTHONPATH
```

**Port conflicts:** If you see `OSError: Cannot find empty port in range: 7860-7860`,
a previous app session is still running. Kill it with:
```bash
fuser -k 7860/tcp 2>/dev/null
```
Then rerun `python src/app.py`.

**API key:** Never paste your API key into a chat window or commit it to GitHub.
Only set it in `~/.bashrc` or directly in the terminal as an environment variable.

---

## Workflow

```
python src/knowledge_base.py    ← once only (builds ChromaDB)
        ↓
python src/app.py               ← Tab 1: enter patient → generate plan
        ↓                       ← Tab 2: review each item (approve/modify/reject)
   (repeat 10×+)                ← Tab 3: watch reward analytics update live
        ↓
python src/train.py             ← offline RL training + evaluation plots
```

---

## Project structure

```
~/integrative_rl/
├── src/
│   ├── config.py           # Paths, API key, reward weights, feature lists
│   ├── knowledge_base.py   # ChromaDB vector store: build + retrieve
│   ├── episode_store.py    # SQLite persistence + reward computation
│   ├── plan_generator.py   # RAG prompt builder + Claude API call
│   ├── app.py              # Unified Gradio app (Intake / Review / Dashboard)
│   └── train.py            # Offline RL training + evaluation
├── data/
│   ├── chroma_db/          # Persistent vector store (auto-created)
│   ├── episodes/           # Draft JSON + rl_episodes.db
│   └── figures/            # Saved evaluation plots
├── models/
│   ├── reward_predictor.pt # Trained policy weights
│   └── metrics.json        # Latest evaluation metrics
├── requirements.txt
└── README.md
```

All data in `~/integrative_rl/data/` is **persistent** across OOD sessions.

---

## After collecting 10+ episodes — run RL training

```bash
cd ~/integrative_rl
python src/train.py
```

Output: `models/reward_predictor.pt` + `data/figures/policy_evaluation.png`

---

## Tuning the reward function

Edit `src/config.py` → `REWARD_CONFIG`:

| Key | Default | Effect |
|---|---|---|
| `approve` | +1.0 | Base reward for approval |
| `modify` | +0.3 | Partial credit |
| `reject` | -0.5 | Penalty |
| `cross_synergy_bonus` | +0.25 | Reward cross-tradition convergence |
| `evidence_A_bonus` | +0.20 | Prefer RCT-backed recommendations |
| `herb_drug_penalty` | -0.80 | Safety constraint |
| `high_priority_weight` | ×1.3 | Scale rewards for high-priority items |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: gradio` | `python -m pip install --user gradio` |
| `No module named 'orjson.orjson'` | `rm -rf ~/.local/lib/python3.11` then reinstall |
| `OSError: port 7860 in use` | `fuser -k 7860/tcp 2>/dev/null` |
| `ANTHROPIC_API_KEY not set` | `export ANTHROPIC_API_KEY=sk-ant-...` |
| Packages installed but not found | `export PYTHONPATH=$HOME/.local/lib/python3.13/site-packages:$PYTHONPATH` |
| `module load anaconda` fails | Don't use module load — system Python works directly |
