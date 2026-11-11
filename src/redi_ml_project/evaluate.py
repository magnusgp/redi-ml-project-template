"""Explicit final evaluation on the untouched test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from .data import assert_disjoint_splits, load_data


def final_test(artifact_dir: Path = Path("artifacts"), *, allow_repeat: bool = False) -> dict:
    validation_report = artifact_dir / "validation_report.json"
    model_file = artifact_dir / "model.joblib"
    output = artifact_dir / "final_test_report.json"
    if not (validation_report.exists() and model_file.exists()):
        raise FileNotFoundError("Run training first.")
    if output.exists() and not allow_repeat:
        raise RuntimeError("Final test has already been evaluated. Repeated peeking can bias your decisions.")
    data = load_data()
    assert_disjoint_splits(data)
    model = joblib.load(model_file)  # only load models you created and trust
    X_test = data.X[data.test_ids]
    y_test = data.y[data.test_ids]
    prediction = model.predict(X_test)
    report = {
        "test_n": len(y_test),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=list(range(10))).tolist(),
        "important_limitation": "Random split from digit corpus; cannot establish new-writer or phone-camera performance.",
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run final held-out test (explicit confirmation required)")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if not args.final_test:
        parser.error("Supply --final-test after all model and metric decisions are frozen.")
    print(json.dumps(final_test(args.artifact_dir), indent=2))


if __name__ == "__main__":
    main()
