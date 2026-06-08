Artifacts and outputs
=====================

Each training run writes a timestamped directory under ``artifacts/runs/``. The
project also maintains a lightweight registry at ``artifacts/registry.json``.

Run directory layout
--------------------

A typical run directory looks like this:

.. code-block:: text

   artifacts/runs/<run-name>/
   ├── model.joblib | model.pt
   ├── metrics.json
   ├── metadata.json
   ├── config.resolved.yaml
   ├── fold_metrics.json              # optional
   ├── feature_importance.csv         # optional
   └── MODEL_CARD.md                  # optional, created by bc-mlops report

Required outputs from training
------------------------------

``model.joblib`` or ``model.pt``
   Serialized model artifact written by the selected backend.

``metrics.json``
   Scalar evaluation metrics used by validation, dashboards, and reports.

``metadata.json``
   Run metadata including dataset, evaluation mode, model runtime, config
   snapshot, and MLflow identifiers.

``config.resolved.yaml``
   Fully resolved training config written for reproducibility.

Optional outputs
----------------

``fold_metrics.json``
   Written when ``evaluation.mode: stratified_k_fold`` is used. Contains per-fold
   metrics and summary statistics such as mean/std.

``feature_importance.csv``
   Written when the backend emits a feature-importance view.

   - logistic regression: coefficient-based ranking
   - random forest: tree feature importances
   - PyTorch MLP: first-layer weight magnitude summary
   - histogram gradient boosting: not currently emitted

``MODEL_CARD.md``
   Written by ``bc-mlops report`` after training. Training itself does **not**
   automatically create model cards.

Registry file
-------------

``artifacts/registry.json`` is a lightweight index of completed runs. It stores a
small summary payload used by ``compare`` and the dashboard.

Do not treat it as a complete metadata mirror of the run directory. If you need
rich detail, inspect ``metadata.json`` or the MLflow run.

Quality gates
-------------

Validation reads:

- ``metrics.json`` from a run directory
- ``configs/quality_gates.yaml`` for thresholds

Example:

.. code-block:: bash

   uv run bc-mlops validate \
     --metrics artifacts/runs/<run-name>/metrics.json \
     --gates configs/quality_gates.yaml

Inference inputs
----------------

Offline inference uses a saved model artifact plus JSON or CSV input records:

.. code-block:: bash

   uv run bc-mlops predict \
     --model artifacts/runs/<run-name>/model.joblib \
     --input sample-inputs/sample.json

See :doc:`howtos/run-inference` for format details.

Model cards
-----------

Generate a model card after training:

.. code-block:: bash

   uv run bc-mlops report \
     --run-dir artifacts/runs/<run-name> \
     --output artifacts/runs/<run-name>/MODEL_CARD.md

For k-fold runs, the model card includes a compact cross-validation summary based
on ``fold_metrics.json`` when that file exists.
