"""Train only on training rows; validate on validation rows; never inspect test scores."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import SEED, assert_disjoint_splits, load_data


def build_model():
    # Scaler is FIT inside the pipeline on each training fold, never on held-out rows.
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))


def train(artifact_dir: Path = Path("artifacts")) -> dict:
    data = load_data()
    assert_disjoint_splits(data)
    X_train, y_train = data.X[data.train_ids], data.y[data.train_ids]
    X_val, y_val = data.X[data.validation_ids], data.y[data.validation_ids]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(
        build_model(), X_train, y_train, cv=cv, scoring="balanced_accuracy"
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    baseline_predictions = baseline.predict(X_val)

    report = {
        "data": "scikit-learn load_digits (real 8x8 images, 1797 samples)",
        "seed": SEED,
        "population_claim": "Similar normalized 8x8 digit images from the same source corpus only; no new-writer claim.",
        "split_sizes": {
            "train": int(len(data.train_ids)),
            "validation": int(len(data.validation_ids)),
            "test_reserved": int(len(data.test_ids)),
        },
        "validation_metrics": {
            "baseline_balanced_accuracy": float(balanced_accuracy_score(y_val, baseline_predictions)),
            "model_balanced_accuracy": float(balanced_accuracy_score(y_val, predictions)),
            "model_accuracy": float(accuracy_score(y_val, predictions)),
            "confusion_matrix": confusion_matrix(y_val, predictions, labels=list(range(10))).tolist(),
        },
        "train_only_cv": {
            "folds": 5,
            "balanced_accuracy_mean": float(cv_scores.mean()),
            "balanced_accuracy_std": float(cv_scores.std()),
        },
        "test_policy": "Not evaluated in train.py. Run evaluate.py --final-test ONCE after decisions are frozen.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "model.joblib")
    (artifact_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Validation balanced accuracy:", round(report["validation_metrics"]["model_balanced_accuracy"], 3))
    print("Held-out test score: NOT COMPUTED")
    return report


def main():
    parser = argparse.ArgumentParser(description="Train and validate without touching the test score")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    train(args.artifact_dir)


if __name__ == "__main__":
    main()
