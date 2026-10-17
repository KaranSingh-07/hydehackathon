"""Dataset Audit, Model Training, and Evaluation for Learned Catan Value Functions.

Audits dataset integrity, trains compact linear and shallow MLP value models,
evaluates out-of-sample prediction quality, and exports pure-NumPy inference artifacts.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure project root is in sys.path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from my_team.evaluation.features import extract_strategic_features


FEATURE_NAMES = [
    "is_winning",
    "my_actual_vps",
    "opp_public_vps",
    "vp_delta",
    "my_dim_production",
    "opp_dim_production",
    "production_diversity",
    "wheat_ore_pips",
    "wood_brick_pips",
    "has_3_to_1_port",
    "has_special_port",
    "port_synergy",
    "can_city_readiness",
    "can_settlement_readiness",
    "can_dev_readiness",
    "discard_risk",
    "num_buildable_nodes",
    "blocked_penalty",
    "my_road_length",
    "has_longest_road",
    "road_lead",
    "my_knights_played",
    "has_largest_army",
    "army_lead",
    "unplayed_devs",
    "robber_on_enemy",
    "robber_on_self",
]


def load_dataset_records(jsonl_paths: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load and parse records from JSONL files, extracting header metadata."""
    records = []
    metadata = {}
    for p in jsonl_paths:
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "_metadata" in obj:
                    metadata.update(obj["_metadata"])
                else:
                    records.append(obj)
    return records, metadata


def audit_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform rigorous audit of data volume, balance, phases, and leakage."""
    total_decisions = len(records)
    if total_decisions == 0:
        return {"error": "Dataset is empty."}

    games = set(r["game_id"] for r in records)
    seeds = set(r["board_seed"] for r in records)

    outcomes = {"WIN": 0, "LOSS": 0, "DRAW": 0}
    phases = {"EARLY": 0, "MID": 0, "LATE": 0}
    players = {"A": 0, "B": 0}

    missing_features_count = 0
    all_features_matrix = []

    for r in records:
        outcomes[r.get("outcome", "DRAW")] = outcomes.get(r.get("outcome", "DRAW"), 0) + 1
        phases[r.get("phase", "MID")] = phases.get(r.get("phase", "MID"), 0) + 1
        players[r.get("player_label", "A")] = players.get(r.get("player_label", "A"), 0) + 1

        f_dict = r.get("features", {})
        row = []
        for feat in FEATURE_NAMES:
            if feat not in f_dict:
                missing_features_count += 1
                row.append(0.0)
            else:
                row.append(float(f_dict[feat]))
        all_features_matrix.append(row)

    feat_arr = np.array(all_features_matrix, dtype=np.float32)
    feat_means = np.mean(feat_arr, axis=0)
    feat_stds = np.std(feat_arr, axis=0)

    feature_summary = {
        name: {"mean": round(float(m), 3), "std": round(float(s), 3)}
        for name, m, s in zip(FEATURE_NAMES, feat_means, feat_stds)
    }

    audit_report = {
        "total_decisions": total_decisions,
        "unique_games": len(games),
        "unique_seeds": len(seeds),
        "outcome_distribution": {k: f"{v} ({v/total_decisions*100:.1f}%)" for k, v in outcomes.items()},
        "phase_distribution": {k: f"{v} ({v/total_decisions*100:.1f}%)" for k, v in phases.items()},
        "policy_distribution": {k: f"{v} ({v/total_decisions*100:.1f}%)" for k, v in players.items()},
        "missing_values": missing_features_count,
        "feature_summary": feature_summary,
    }
    return audit_report


def split_dataset_by_seeds(
    records: List[Dict[str, Any]],
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition records strictly by board_seed to prevent intra-game observation leakage."""
    all_seeds = sorted(list(set(r["board_seed"] for r in records)))
    rng = random.Random(seed)
    rng.shuffle(all_seeds)

    n_val = max(1, int(len(all_seeds) * val_ratio))
    val_seeds = set(all_seeds[:n_val])
    train_seeds = set(all_seeds[n_val:])

    train_records = [r for r in records if r["board_seed"] in train_seeds]
    val_records = [r for r in records if r["board_seed"] in val_seeds]

    # Verification: Confirm zero seed overlap
    assert len(train_seeds.intersection(val_seeds)) == 0, "Seed leakage detected!"
    return train_records, val_records


def extract_X_y(records: List[Dict[str, Any]], target: str = "is_win") -> Tuple[np.ndarray, np.ndarray]:
    """Extract design matrix X and target vector y from parsed records."""
    X = []
    y = []
    for r in records:
        f_dict = r.get("features", {})
        row = [float(f_dict.get(k, 0.0)) for k in FEATURE_NAMES]
        X.append(row)

        if target == "is_win":
            y.append(float(r.get("is_win", 0.0)))
        elif target == "vps":
            y.append(float(r.get("final_vps_me", 0.0)) / 10.0)
        elif target == "delta":
            y.append(float(r.get("vps_delta", 0.0)) / 10.0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_linear_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    """Train Ridge regression value model."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, roc_auc_score

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    train_mse = float(mean_squared_error(y_train, train_preds))
    val_mse = float(mean_squared_error(y_val, val_preds))

    # Clip for AUC if binary win target
    train_auc = float(roc_auc_score(y_train, np.clip(train_preds, 0, 1))) if len(set(y_train)) > 1 else 0.5
    val_auc = float(roc_auc_score(y_val, np.clip(val_preds, 0, 1))) if len(set(y_val)) > 1 else 0.5

    results = {
        "train_mse": round(train_mse, 4),
        "val_mse": round(val_mse, 4),
        "train_auc": round(train_auc, 4),
        "val_auc": round(val_auc, 4),
        "weights": model.coef_.tolist(),
        "intercept": float(model.intercept_),
    }
    return model, results


def train_mlp_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    """Train shallow MLP (25 -> 32 -> 1) value model."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error, roc_auc_score

    mlp = MLPRegressor(
        hidden_layer_sizes=(32,),
        activation="relu",
        solver="adam",
        alpha=0.01,
        max_iter=300,
        random_state=42,
        early_stopping=True,
    )
    mlp.fit(X_train, y_train)

    train_preds = mlp.predict(X_train)
    val_preds = mlp.predict(X_val)

    train_mse = float(mean_squared_error(y_train, train_preds))
    val_mse = float(mean_squared_error(y_val, val_preds))

    train_auc = float(roc_auc_score(y_train, np.clip(train_preds, 0, 1))) if len(set(y_train)) > 1 else 0.5
    val_auc = float(roc_auc_score(y_val, np.clip(val_preds, 0, 1))) if len(set(y_val)) > 1 else 0.5

    results = {
        "train_mse": round(train_mse, 4),
        "val_mse": round(val_mse, 4),
        "train_auc": round(train_auc, 4),
        "val_auc": round(val_auc, 4),
        "coefs": [w.tolist() for w in mlp.coefs_],
        "intercepts": [b.tolist() for b in mlp.intercepts_],
    }
    return mlp, results


def export_learned_model(linear_res: Dict[str, Any], mlp_res: Dict[str, Any], filepath: str):
    """Export compact, standalone NumPy-compatible model representation."""
    payload = {
        "schema_version": "1.0.0",
        "feature_names": FEATURE_NAMES,
        "linear_model": {
            "weights": linear_res["weights"],
            "intercept": linear_res["intercept"],
            "val_mse": linear_res["val_mse"],
            "val_auc": linear_res["val_auc"],
        },
        "mlp_model": {
            "weights": mlp_res["coefs"],
            "biases": mlp_res["intercepts"],
            "val_mse": mlp_res["val_mse"],
            "val_auc": mlp_res["val_auc"],
        },
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f" Learned model exported to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Audit Catan dataset and train learned value models.")
    parser.add_argument("--data", type=str, nargs="+", default=["results/datasets/decisions_phase5_train_*.jsonl"], help="Dataset files or globs")
    parser.add_argument("--out", type=str, default="my_team/evaluation/learned_model.json", help="Output model path")
    args = parser.parse_args()

    import glob
    resolved_paths = []
    for pattern in args.data:
        resolved_paths.extend(glob.glob(pattern))

    if not resolved_paths:
        print("No dataset files found matching pattern.")
        return

    print(f"Loading {len(resolved_paths)} dataset files...")
    records, metadata = load_dataset_records(resolved_paths)
    print(f"Loaded {len(records)} decision records.")

    print("\n================ DATASET AUDIT ================")
    audit = audit_dataset(records)
    print(f" Total Decisions:       {audit['total_decisions']}")
    print(f" Unique Games:          {audit['unique_games']}")
    print(f" Unique Board Seeds:    {audit['unique_seeds']}")
    print(f" Outcome Distribution:  {audit['outcome_distribution']}")
    print(f" Phase Distribution:    {audit['phase_distribution']}")
    print(f" Policy Distribution:   {audit['policy_distribution']}")
    print(f" Missing Features:      {audit['missing_values']}")
    print("===============================================\n")

    train_recs, val_recs = split_dataset_by_seeds(records, val_ratio=0.25)
    print(f"Split by seed: {len(train_recs)} train decisions, {len(val_recs)} validation decisions.")

    X_train, y_train = extract_X_y(train_recs, target="is_win")
    X_val, y_val = extract_X_y(val_recs, target="is_win")

    print("\nTraining Linear Value Model (Ridge)...")
    lin_model, lin_res = train_linear_model(X_train, y_train, X_val, y_val)
    print(f" Linear -> Train MSE: {lin_res['train_mse']}, Val MSE: {lin_res['val_mse']} | Train AUC: {lin_res['train_auc']}, Val AUC: {lin_res['val_auc']}")

    print("\nTraining Shallow MLP Value Model (25 -> 32 -> 1)...")
    mlp_model, mlp_res = train_mlp_model(X_train, y_train, X_val, y_val)
    print(f" MLP    -> Train MSE: {mlp_res['train_mse']}, Val MSE: {mlp_res['val_mse']} | Train AUC: {mlp_res['train_auc']}, Val AUC: {mlp_res['val_auc']}")

    export_learned_model(lin_res, mlp_res, args.out)


if __name__ == "__main__":
    main()
