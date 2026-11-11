# Example model card — handwritten digits (teaching only)

**Name:** ReDI 8x8 Digit Demo v0.2. This is an educational classifier.

**Intended use:** Demonstrate a repeatable ML workflow and predict labels for normalized
8x8 digit images similar to the scikit-learn `load_digits()` corpus.

**Not intended for:** Reading identity documents, recognising digits from smartphone photos,
real-world handwriting services, or making any important decisions.

**Population and source:** 1,797 8x8 digit images embedded in scikit-learn. The source data
was derived from historical forms; writer identities are not present in `load_digits()`.

**Inputs and target:** 64 image intensity features (values 0–16), digit class 0–9.

**Splits:** Stratified random 60%/20%/20% train/validation/test, fixed seed 42.
The split proves row-level separation, **not** new-writer independence.

**Leakage prevention:** Split before learning scaling parameters; scaler is within an
`sklearn.pipeline.Pipeline`; CV operates on training rows only. Final test requires explicit
separate command and should not guide changes.

**Baseline and results:** The training script reports a majority-class baseline and
classification metrics on validation, plus 5-fold CV on training. For the demonstration run (scikit-learn 1.8, seed 42): baseline balanced accuracy **0.100**, model validation balanced accuracy **0.972**, and training-only 5-fold CV balanced accuracy **0.967 ± 0.012**. The reserved final test was not evaluated. These are in-corpus figures, not real-world deployment evidence. Final test is reported only after choices are frozen.

**Limitations:** Small, normalized historical digit images; unknown writer distribution; no
external performance validation. Demo has no monitoring of incoming real-world traffic.

**Operation:** Run `uv run python -m redi_ml_project.train` and optionally
`uv run --extra app streamlit run src/redi_ml_project/app.py`.
