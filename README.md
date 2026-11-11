# ReDI ML Project Starter — Digit recogniser

A small, working example of **image classification turned into an application**. It recognises numbers from 1,797 real 8×8 handwritten-digit images bundled with scikit-learn. The larger **MNIST** dataset is a separate 28×28 dataset; this starter uses the small digits dataset so the first run works locally.

## Run the project

Install [uv](https://docs.astral.sh/uv/) and Python 3.11 or newer. From the repository directory:

```bash
uv sync --extra dev --extra app
uv run python -m redi_ml_project.train
uv run pytest -q
uv run streamlit run src/redi_ml_project/app.py
```

The first run needs internet for package installation. The data itself loads locally from scikit-learn. Model artifacts under `artifacts/` are generated on your computer and excluded from Git.

The app lets you select an image, make a prediction, inspect predicted probabilities, and reveal the correct label. It uses validation images, not the final test set.

## Use this as your project starting point

1. Write down your **problem**, **target users**, and **data source** in `docs/PROJECT_CHARTER.md`.
2. Replace `src/redi_ml_project/data.py` with your own permitted dataset and appropriate split.
3. Develop a simple **baseline** and an evaluation suited to the task.
4. Update the **model card** with intended use, results, and limitations.
5. Adapt the optional Streamlit app or choose a simpler presentation format.
6. Create small GitHub Issues and review one another's code.

For a GitHub template, the teacher creates a public repository and checks *Template repository* in GitHub settings. Each group can then use **Use this template** to create its own independent repository.

## What's inside?

- `src/redi_ml_project/train.py` — simple classifier, training, validation and training-only CV
- `src/redi_ml_project/evaluate.py` — separate final test evaluation, after decisions are fixed
- `src/redi_ml_project/app.py` — interactive prediction app
- `tests/` — automatic split and training checks
- `docs/PROJECT_CHARTER.md` — planning prompts
- `docs/MODEL_CARD.md` — intended users, evidence and limitations
- `.github/` — basic Issues, Pull Request and GitHub Actions templates

## Limits of the example

The demonstration uses a random split within the historical source dataset. It does not establish generalisation to phone-camera images, different handwriting conditions, or wholly new writers. The reserved final test is intentionally kept separate from model development.

Only use data that your team has permission to use. Do not commit personal data, passwords or private files to GitHub.

## Sources

- [scikit-learn Digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [MNIST dataset documentation](https://keras.io/api/datasets/mnist/)
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
