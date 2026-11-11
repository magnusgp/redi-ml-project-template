"""Real optical digit images; stable, explicitly disjoint splits."""
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42

@dataclass(frozen=True)
class DigitData:
    X: np.ndarray
    y: np.ndarray
    train_ids: np.ndarray
    validation_ids: np.ndarray
    test_ids: np.ndarray


def load_data(seed: int = SEED) -> DigitData:
    digits = load_digits()
    X = np.asarray(digits.data)
    y = np.asarray(digits.target)
    ids = np.arange(len(y))
    development_ids, test_ids = train_test_split(
        ids, test_size=0.20, stratify=y, random_state=seed
    )
    train_ids, validation_ids = train_test_split(
        development_ids,
        test_size=0.25,  # 60% train, 20% validation, 20% test overall
        stratify=y[development_ids],
        random_state=seed,
    )
    return DigitData(X, y, np.sort(train_ids), np.sort(validation_ids), np.sort(test_ids))


def assert_disjoint_splits(data: DigitData) -> None:
    train = set(data.train_ids.tolist())
    val = set(data.validation_ids.tolist())
    test = set(data.test_ids.tolist())
    assert not (train & val), "Training and validation overlap"
    assert not (train & test), "Training and test overlap"
    assert not (val & test), "Validation and test overlap"
    assert len(train | val | test) == len(data.y), "Some samples are missing"
