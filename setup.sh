#!/bin/bash
# setup.sh — One-time setup for Open OnDemand / UVA Rivanna JupyterLab
# Run this once from a JupyterLab terminal:
#   bash ~/integrative_rl/setup.sh

set -e
echo "=== Integrative Medicine HITL-RL — OOD Setup ==="

# ── Load the Anaconda module (UVA Rivanna)
module load anaconda 2>/dev/null || module load miniforge 2>/dev/null || \
    echo "Note: no anaconda module found — assuming conda is already in PATH"

# ── Create conda environment
echo ""
echo "Creating conda environment 'integrative_rl'..."
conda create -y -n integrative_rl python=3.11 2>/dev/null || \
    echo "Environment already exists, skipping create."

# ── Activate and install
echo "Installing dependencies..."
conda run -n integrative_rl pip install -r ~/integrative_rl/requirements.txt

# ── Build ChromaDB vector store
echo ""
echo "Building knowledge vector store (~30 seconds)..."
conda run -n integrative_rl python ~/integrative_rl/src/knowledge_base.py

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Set API key (paste into ~/.bashrc for persistence):"
echo "     export ANTHROPIC_API_KEY=sk-ant-..."
echo ""
echo "  2. Launch the app from a JupyterLab terminal:"
echo "     module load anaconda"
echo "     conda activate integrative_rl"
echo "     cd ~/integrative_rl"
echo "     python src/app.py"
echo ""
echo "  3. Open the app in your browser:"
echo "     In OOD, your session URL looks like:"
echo "     https://ood.hpc.virginia.edu/node/udc-an26-1/34567/..."
echo "     Change it to:  https://ood.hpc.virginia.edu/node/udc-an26-1/7860/"
echo "     (swap the JupyterLab port with 7860)"
echo ""
echo "  4. After 10+ reviewed episodes, run RL training:"
echo "     python ~/integrative_rl/src/train.py"
