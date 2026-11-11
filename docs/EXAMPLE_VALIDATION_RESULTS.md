# Example validation results (reproducible, not test)

- 1,797 source images: 1,077 train, 360 validation, 360 reserved test.
- Most-frequent baseline balanced accuracy on validation: **0.100**.
- Logistic regression pipeline balanced accuracy on validation: **0.972**.
- 5-fold balanced accuracy *on training only*: **0.967 ± 0.012**.
- Final test: **not evaluated**.
- Data distribution limitation: row-level split does not establish writer-level or camera-photo generalisation.

Recreate with `uv run python -m redi_ml_project.train`.
