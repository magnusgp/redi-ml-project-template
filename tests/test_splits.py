import numpy as np

from redi_ml_project.data import assert_disjoint_splits, load_data


def test_splits_are_disjoint_and_complete():
    data = load_data()
    assert_disjoint_splits(data)
    assert len(data.train_ids) + len(data.validation_ids) + len(data.test_ids) == 1797


def test_splits_reproduce_with_fixed_seed():
    first, second = load_data(), load_data()
    assert np.array_equal(first.train_ids, second.train_ids)
    assert np.array_equal(first.validation_ids, second.validation_ids)
    assert np.array_equal(first.test_ids, second.test_ids)


def test_test_ids_are_never_in_train_or_validation():
    data = load_data()
    assert not set(data.test_ids) & set(data.train_ids)
    assert not set(data.test_ids) & set(data.validation_ids)
