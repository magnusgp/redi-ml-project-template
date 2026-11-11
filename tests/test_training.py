import pytest
from sklearn.pipeline import Pipeline

from redi_ml_project.evaluate import final_test
from redi_ml_project.train import build_model, train


def test_preprocessing_is_inside_train_only_pipeline():
    assert isinstance(build_model(), Pipeline)
    assert "standardscaler" in build_model().named_steps


def test_training_reports_validation_but_no_test_score(tmp_path):
    result = train(tmp_path)
    assert result["split_sizes"]["test_reserved"] > 0
    assert "test_metrics" not in result
    assert "final_test_report" not in [p.stem for p in tmp_path.iterdir()]


def test_final_test_is_not_run_by_automated_checks(tmp_path):
    # CI must not peek at the held-out test. Only students can run the explicit command.
    train(tmp_path)
    assert not (tmp_path / "final_test_report.json").exists()
    with pytest.raises(FileNotFoundError):
        final_test(tmp_path / "not_trained")
