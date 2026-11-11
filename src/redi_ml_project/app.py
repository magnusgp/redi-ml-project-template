"""A small Streamlit app for exploring digit-recognition predictions."""
from pathlib import Path

import joblib

from redi_ml_project.data import load_data


def main() -> None:
    import matplotlib.pyplot as plt
    import streamlit as st

    st.set_page_config(page_title="Digit recogniser | ReDI", layout="centered")
    st.title("Can a computer read this digit?")
    st.write("Try real handwritten images from the validation set and inspect the model's predictions.")

    model_path = Path("artifacts/model.joblib")
    if not model_path.exists():
        st.error("First train the model: uv run python -m redi_ml_project.train")
        st.stop()

    # Only load a model file generated locally by the trusted training command.
    model = joblib.load(model_path)
    dataset = load_data()
    ix = st.slider("Choose an image", 0, len(dataset.validation_ids) - 1, 0)
    row_id = int(dataset.validation_ids[ix])
    image = dataset.X[row_id].reshape(8, 8)
    prediction = int(model.predict(dataset.X[[row_id]])[0])
    probabilities = model.predict_proba(dataset.X[[row_id]])[0]

    left, right = st.columns([1, 1.25])
    with left:
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.imshow(image, cmap="Greys", interpolation="nearest", vmin=0, vmax=16)
        ax.set_axis_off()
        st.pyplot(fig)
        plt.close(fig)
        st.metric("Predicted digit", prediction)
        if st.button("Reveal the correct label"):
            st.write(f"Correct digit: **{int(dataset.y[row_id])}**")

    with right:
        st.write("**Model confidence across the ten classes**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar(range(10), probabilities, color="#2875A9")
        ax.set_xticks(range(10))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Predicted probability")
        ax.set_xlabel("Digit")
        st.pyplot(fig)
        plt.close(fig)

    st.caption(
        "This model was evaluated on similar historical 8×8 images. "
        "Its results do not establish accuracy for phone photos or new writing conditions."
    )


if __name__ == "__main__":
    main()
