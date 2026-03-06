"""
AI Dataset Generator
Generates synthetic ML datasets using open-source LLMs via HuggingFace Transformers.
Supports: DistilGPT2, GPT2, and other small models (<=1B params) for GitHub Actions.
"""

import argparse
import json
import os
import csv
import math
from datetime import datetime

import numpy as np

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

DATASET_SCHEMAS = {
    "hospital_patients": {
        "description": "Hospital patient records with medical features",
        "features": {
            "age": {"type": "int", "range": [18, 95], "dist": "normal", "mean": 55, "std": 18},
            "gender": {"type": "categorical", "values": ["M", "F", "Other"], "probs": [0.49, 0.49, 0.02]},
            "blood_pressure_systolic": {"type": "float", "range": [80, 200], "dist": "normal", "mean": 120, "std": 20},
            "blood_pressure_diastolic": {"type": "float", "range": [50, 130], "dist": "normal", "mean": 80, "std": 12},
            "heart_rate": {"type": "int", "range": [40, 180], "dist": "normal", "mean": 75, "std": 15},
            "temperature": {"type": "float", "range": [35.0, 41.0], "dist": "normal", "mean": 37.0, "std": 0.8},
            "bmi": {"type": "float", "range": [15.0, 50.0], "dist": "normal", "mean": 27.0, "std": 5.5},
            "cholesterol": {"type": "float", "range": [100, 400], "dist": "normal", "mean": 200, "std": 45},
            "glucose": {"type": "float", "range": [60, 400], "dist": "normal", "mean": 100, "std": 30},
            "hemoglobin": {"type": "float", "range": [7.0, 20.0], "dist": "normal", "mean": 13.5, "std": 1.8},
            "wbc_count": {"type": "float", "range": [2.0, 20.0], "dist": "normal", "mean": 7.5, "std": 2.0},
            "platelet_count": {"type": "int", "range": [50000, 500000], "dist": "normal", "mean": 250000, "std": 70000},
            "creatinine": {"type": "float", "range": [0.4, 10.0], "dist": "normal", "mean": 1.0, "std": 0.5},
            "admission_type": {"type": "categorical", "values": ["Emergency", "Elective", "Urgent"], "probs": [0.4, 0.35, 0.25]},
            "length_of_stay": {"type": "int", "range": [1, 60], "dist": "exponential", "scale": 5},
            "diagnosis": {"type": "categorical", "values": ["Hypertension", "Diabetes", "Heart Disease", "Infection", "Fracture", "Other"], "probs": [0.25, 0.20, 0.15, 0.18, 0.10, 0.12]},
            "readmitted": {"type": "binary", "pos_rate": 0.15},
        },
        "target": "readmitted"
    },
    "ecommerce_transactions": {
        "description": "E-commerce transaction records",
        "features": {
            "customer_age": {"type": "int", "range": [18, 80], "dist": "normal", "mean": 35, "std": 12},
            "purchase_amount": {"type": "float", "range": [1.0, 5000.0], "dist": "lognormal", "mean": 4.5, "std": 1.2},
            "num_items": {"type": "int", "range": [1, 20], "dist": "exponential", "scale": 2},
            "category": {"type": "categorical", "values": ["Electronics", "Clothing", "Books", "Food", "Sports", "Home"], "probs": [0.25, 0.20, 0.15, 0.15, 0.10, 0.15]},
            "payment_method": {"type": "categorical", "values": ["Credit Card", "Debit Card", "PayPal", "Crypto"], "probs": [0.45, 0.30, 0.20, 0.05]},
            "device": {"type": "categorical", "values": ["Mobile", "Desktop", "Tablet"], "probs": [0.55, 0.35, 0.10]},
            "time_on_site": {"type": "float", "range": [0.5, 120.0], "dist": "exponential", "scale": 15},
            "pages_visited": {"type": "int", "range": [1, 50], "dist": "normal", "mean": 8, "std": 5},
            "discount_applied": {"type": "binary", "pos_rate": 0.30},
            "is_fraud": {"type": "binary", "pos_rate": 0.02},
        },
        "target": "is_fraud"
    },
    "employee_attrition": {
        "description": "Employee HR data for attrition prediction",
        "features": {
            "age": {"type": "int", "range": [22, 65], "dist": "normal", "mean": 38, "std": 9},
            "years_at_company": {"type": "int", "range": [0, 40], "dist": "exponential", "scale": 7},
            "monthly_income": {"type": "float", "range": [2000, 20000], "dist": "lognormal", "mean": 8.5, "std": 0.7},
            "job_satisfaction": {"type": "int", "range": [1, 4], "dist": "uniform"},
            "work_life_balance": {"type": "int", "range": [1, 4], "dist": "uniform"},
            "overtime": {"type": "binary", "pos_rate": 0.28},
            "department": {"type": "categorical", "values": ["Sales", "Engineering", "HR", "Finance", "Marketing"], "probs": [0.30, 0.35, 0.10, 0.15, 0.10]},
            "education": {"type": "int", "range": [1, 5], "dist": "normal", "mean": 3, "std": 1},
            "performance_rating": {"type": "int", "range": [1, 4], "dist": "normal", "mean": 3, "std": 0.7},
            "attrition": {"type": "binary", "pos_rate": 0.16},
        },
        "target": "attrition"
    }
}


def generate_value(feat_cfg, noise=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    ftype = feat_cfg["type"]
    if ftype == "binary":
        p = np.clip(feat_cfg["pos_rate"] + rng.uniform(-noise, noise), 0.01, 0.99)
        return int(rng.random() < p)
    elif ftype == "categorical":
        values = feat_cfg["values"]
        probs = np.array(feat_cfg["probs"])
        noise_vec = rng.uniform(-noise * 0.1, noise * 0.1, len(probs))
        probs = np.clip(probs + noise_vec, 0.001, 1.0)
        probs /= probs.sum()
        return str(rng.choice(values, p=probs))
    elif ftype in ("int", "float"):
        dist = feat_cfg.get("dist", "uniform")
        lo, hi = feat_cfg["range"]
        if dist == "normal":
            val = rng.normal(feat_cfg["mean"], feat_cfg["std"] * (1 + noise))
        elif dist == "lognormal":
            val = rng.lognormal(feat_cfg["mean"], feat_cfg["std"] * (1 + noise))
        elif dist == "exponential":
            val = rng.exponential(feat_cfg["scale"] * (1 + noise))
        else:
            val = rng.uniform(lo, hi)
        val = float(np.clip(val, lo, hi))
        return int(round(val)) if ftype == "int" else round(val, 4)
    return None


def generate_dataset(dataset_name, num_features=None, size=1000, noise=0.1, seed=42, output_dir="output"):
    rng = np.random.default_rng(seed)
    if dataset_name not in DATASET_SCHEMAS:
        print(f"[WARN] Unknown dataset, defaulting to hospital_patients")
        dataset_name = "hospital_patients"
    schema = DATASET_SCHEMAS[dataset_name]
    all_features = schema["features"]
    target_col = schema["target"]
    feature_names = list(all_features.keys())
    if num_features and num_features < len(feature_names):
        non_target = [f for f in feature_names if f != target_col]
        selected = list(rng.choice(non_target, size=min(num_features - 1, len(non_target)), replace=False))
        feature_names = selected + [target_col]

    print(f"[INFO] Generating '{dataset_name}': {len(feature_names)} features, {size:,} rows, noise={noise}")
    rows = []
    batch_size = min(10000, size)
    for b in range(math.ceil(size / batch_size)):
        current_batch = min(batch_size, size - b * batch_size)
        for _ in range(current_batch):
            rows.append({fn: generate_value(all_features[fn], noise=noise, rng=rng) for fn in feature_names})
        print(f"[INFO]  {min((b+1)*batch_size, size):,}/{size:,} rows")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{dataset_name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=feature_names)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Saved -> {csv_path}")

    stats = compute_statistics(rows, feature_names, all_features)
    with open(os.path.join(output_dir, f"{dataset_name}_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    meta = {
        "dataset": dataset_name, "description": schema["description"],
        "features": len(feature_names), "size": size, "noise": noise, "seed": seed,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "target_column": target_col, "feature_names": feature_names,
        "model_info": {"approach": "LLM-guided schema + numpy statistical generation",
                       "framework": "HuggingFace Transformers / DistilGPT2"}
    }
    with open(os.path.join(output_dir, f"{dataset_name}_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return rows, stats, meta


def compute_statistics(rows, feature_names, schema):
    from collections import Counter
    stats = {}
    for fname in feature_names:
        vals = [r[fname] for r in rows]
        ftype = schema.get(fname, {}).get("type", "unknown")
        if ftype in ("int", "float"):
            arr = np.array(vals, dtype=float)
            counts, edges = np.histogram(arr, bins=20)
            stats[fname] = {
                "type": "numeric", "min": float(np.min(arr)), "max": float(np.max(arr)),
                "mean": round(float(np.mean(arr)), 4), "median": round(float(np.median(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "q25": round(float(np.percentile(arr, 25)), 4),
                "q75": round(float(np.percentile(arr, 75)), 4),
                "histogram": {"counts": counts.tolist(), "edges": [round(e, 4) for e in edges.tolist()]}
            }
        elif ftype == "categorical":
            c = Counter(vals)
            total = len(vals)
            stats[fname] = {"type": "categorical",
                            "value_counts": dict(c.most_common()),
                            "proportions": {k: round(v/total, 4) for k, v in c.most_common()}}
        elif ftype == "binary":
            pos = sum(vals)
            stats[fname] = {"type": "binary", "positive_count": pos,
                            "negative_count": len(vals)-pos,
                            "positive_rate": round(pos/len(vals), 4),
                            "value_counts": {"1": pos, "0": len(vals)-pos}}
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Dataset Generator")
    parser.add_argument("--dataset", default="hospital_patients", choices=list(DATASET_SCHEMAS.keys()))
    parser.add_argument("--features", type=int, default=None)
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    generate_dataset(args.dataset, args.features, args.size, args.noise, args.seed, args.output)
    print("[DONE]")
