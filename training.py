"""
train.py — Offline RL training script. 

Run after collecting ≥10 reviewed episodes.

Usage:
    cd <project_root>
    python src/train.py
    python src/train.py --epochs 200 --min_episodes 5
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for local scripts
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import MODELS_DIR, FIGURES_DIR, SYSTEMS, CATEGORIES, PRIORITIES, EVIDENCE
from episode_store import load_all_episodes, episode_count


# ── Feature encoding

def encode_features(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for _, row in df.iterrows():
        vec  = [1.0 if row["system"]         == s else 0.0 for s in SYSTEMS]
        vec += [1.0 if row["category"]       == c else 0.0 for c in CATEGORIES]
        vec += [1.0 if row["priority"]       == p else 0.0 for p in PRIORITIES]
        vec += [1.0 if row["evidence_grade"] == e else 0.0 for e in EVIDENCE]
        vec += [float(row.get("has_cross_synergy", 0)),
                float(row.get("has_herb_drug_flag", 0))]
        rows.append(vec)
    return np.array(rows, dtype=np.float32)


# ── Policy model

class RewardPredictor(nn.Module):
    """MLP: item feature vector → expected clinician reward."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_policy(X_tr, y_tr, X_val, y_val, epochs=150, lr=1e-3, batch_size=16, patience=25):
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    model   = RewardPredictor(X_tr.shape[1]).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_tr).to(device)
    yt = torch.tensor(y_tr).to(device)
    Xv = torch.tensor(X_val).to(device)
    yv = torch.tensor(y_val).to(device)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    tr_hist, va_hist = [], []
    best_val, best_state, no_imp = np.inf, None, 0

    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        tr_loss = ep_loss / len(loader)

        model.eval()
        with torch.no_grad():
            va_loss = loss_fn(model(Xv), yv).item()

        tr_hist.append(tr_loss)
        va_hist.append(va_loss)
        sched.step(va_loss)

        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1

        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch+1:3d} | train MSE: {tr_loss:.4f} | val MSE: {va_loss:.4f}")

        if no_imp >= patience:
            print(f"  Early stop @ epoch {epoch+1}  (best val MSE: {best_val:.4f})")
            break

    model.load_state_dict(best_state)
    return model, tr_hist, va_hist


def reward_to_action(r: float) -> str:
    if r >= 0.7:   return "approve"
    elif r >= 0.0: return "modify"
    else:          return "reject"


def evaluate(model, X_test, y_test, test_items, tr_hist, va_hist):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test).to(device)).cpu().numpy()

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print("\n=== Reward Prediction Metrics (test set) ===")
    for name, val in [("MSE", mse), ("RMSE", rmse), ("MAE", mae), ("R²", r2)]:
        print(f"  {name}: {val:.4f}")

    y_true_actions = test_items["action"].values
    y_pred_actions = np.array([reward_to_action(r) for r in y_pred])
    labels         = ["approve", "modify", "reject"]

    print("\n=== Action Classification (test set) ===")
    print(classification_report(y_true_actions, y_pred_actions, labels=labels, zero_division=0))

    baseline_rate = (y_true_actions == "approve").mean()
    policy_rate   = (y_pred_actions == "approve").mean()
    sorted_idx    = np.argsort(y_pred)[::-1]
    topk_rate     = (y_true_actions[sorted_idx[: len(y_pred) // 2]] == "approve").mean()

    print("=== Primary RL metrics ===")
    print(f"  Clinician approval rate (actual):          {baseline_rate*100:.1f}%")
    print(f"  Policy-predicted approval rate:            {policy_rate*100:.1f}%")
    print(f"  Top-50% by predicted reward → approval:   {topk_rate*100:.1f}%  "
          f"(delta: {(topk_rate-baseline_rate)*100:+.1f}%)")

    # ── Save evaluation figures
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plt.rcParams.update({"font.size": 11})

    axes[0,0].plot(tr_hist, label="Train MSE", color="#1D9E75")
    axes[0,0].plot(va_hist, label="Val MSE",   color="#D85A30", linestyle="--")
    axes[0,0].set_title("A — Training & validation loss")
    axes[0,0].legend(frameon=False)

    axes[0,1].scatter(y_test, y_pred, alpha=0.6, color="#7F77DD", s=50, edgecolors="white")
    lims = [min(y_test.min(), y_pred.min()) - 0.1, max(y_test.max(), y_pred.max()) + 0.1]
    axes[0,1].plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
    axes[0,1].set_title(f"B — Predicted vs actual (R²={r2:.3f})")
    axes[0,1].set_xlabel("Actual"); axes[0,1].set_ylabel("Predicted")

    cm = confusion_matrix(y_true_actions, y_pred_actions, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=axes[1,0], colorbar=False, cmap="Greens")
    axes[1,0].set_title("C — Action confusion matrix")

    cum = np.cumsum(y_true_actions[sorted_idx] == "approve") / np.arange(1, len(y_pred) + 1)
    axes[1,1].plot(range(1, len(y_pred) + 1), cum * 100, color="#1D9E75")
    axes[1,1].axhline(baseline_rate * 100, color="#888780", linestyle="--", label="Baseline")
    axes[1,1].set_title("D — Precision@k")
    axes[1,1].set_xlabel("Top-k items"); axes[1,1].set_ylabel("Approval rate (%)")
    axes[1,1].legend(frameon=False)

    fig.suptitle("Policy evaluation — held-out test set", y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "policy_evaluation.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved → {out}")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2,
            "baseline_approval": baseline_rate, "topk_approval": topk_rate}


def main(epochs=150, min_episodes=10):
    n = episode_count()
    if n < min_episodes:
        print(f"Only {n} episodes in store (need {min_episodes}). "
              "Run the app and complete more clinician reviews first.")
        return

    print(f"Loading {n} episodes...")
    items_df, eps_df = load_all_episodes()
    print(f"Item reviews: {len(items_df)}")

    # Preprocessing
    items_df["clinician_note"]  = items_df["clinician_note"].fillna("")
    items_df["evidence_grade"]  = items_df["evidence_grade"].fillna("C – traditional use only")
    items_df["system"]          = items_df["system"].fillna("Integrated")
    items_df["category"]        = items_df["category"].fillna("Lifestyle medicine")
    items_df["priority"]        = items_df["priority"].fillna("Medium")

    # Episode-level train/test split (stratified by reward quartile)
    eps_df["reward_quartile"] = pd.qcut(
    eps_df["reward_total"], q=2, labels=False, duplicates="drop"
)
    train_ids, test_ids = train_test_split(
    eps_df["episode_id"], test_size=0.20, random_state=42,
    stratify=eps_df["reward_quartile"],
)
    train_items = items_df[items_df["episode_id"].isin(train_ids)].copy()
    test_items  = items_df[items_df["episode_id"].isin(test_ids)].copy()

    print(f"Train: {len(train_ids)} eps ({len(train_items)} items)  |  "
          f"Test: {len(test_ids)} eps ({len(test_items)} items)")

    X_train = encode_features(train_items)
    y_train = train_items["item_reward"].values.astype(np.float32)
    X_test  = encode_features(test_items)
    y_test  = test_items["item_reward"].values.astype(np.float32)

    print(f"Feature dim: {X_train.shape[1]}")

    model, tr_hist, va_hist = train_policy(X_train, y_train, X_test, y_test, epochs=epochs)

    # Save model
    out = MODELS_DIR / "reward_predictor.pt"
    torch.save(model.state_dict(), out)
    print(f"\nModel saved → {out}")

    # Evaluate
    metrics = evaluate(model, X_test, y_test, test_items, tr_hist, va_hist)

    # Save metrics
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved → models/metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--min_episodes", type=int, default=10)
    args = parser.parse_args()
    main(args.epochs, args.min_episodes)
