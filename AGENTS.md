# Guidelines for coding assistants

Help the student understand each change. Do not silently choose the target population, data split,
or evaluation metric. Explain these decisions and confirm them with the team.

- Keep training and final test separate. Never evaluate the reserved test while tuning.
- Preserve the preprocessing pipeline so it is fitted on training rows only.
- After coherent changes, run tests and commit code/config/docs locally (no attribution trailers).
- Do not push, merge, publish, or deploy without explicit student approval.
- Do not commit private data, large datasets, models, credentials, or generated outputs.
- Document risks and unsupported claims in docs/MODEL_CARD.md.
- Prefer small changes the students can explain at the final presentation.
