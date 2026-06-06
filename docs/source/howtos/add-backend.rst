How to add a new backend
========================

This repository is designed so model swaps happen in configuration rather than
through CLI churn.

1. Extend configuration defaults
--------------------------------

Add a new entry under ``DEFAULT_MODEL_PARAMS`` in
``src/bc_mlops_showcase/config.py`` and ensure ``_resolve_model_config()`` can
validate the new ``model.kind``.

2. Implement training logic
---------------------------

Add a backend implementation in ``src/bc_mlops_showcase/modeling.py`` that can:

- train from pandas features and labels
- return a ``BackendTrainingBundle``
- persist a portable artifact
- predict positive-class probabilities during inference

3. Register the backend
-----------------------

Update ``train_backend()`` so that the new ``model.kind`` maps to your trainer.

4. Add a config example
-----------------------

Create a YAML config under ``configs/`` showing the new backend's
hyperparameters.

5. Verify the full lifecycle
----------------------------

Run the same contract as existing backends:

.. code-block:: bash

   uv run bc-mlops train --config configs/<your-backend>.yaml --output-dir artifacts/runs
   uv run bc-mlops validate --metrics artifacts/runs/<run-name>/metrics.json --gates configs/quality_gates.yaml
   uv run bc-mlops predict --model artifacts/runs/<run-name>/<artifact> --input sample-inputs/sample.json
   uv run bc-mlops report --run-dir artifacts/runs/<run-name> --output artifacts/runs/<run-name>/MODEL_CARD.md

The goal is consistency: backend-specific implementation, shared pipeline
contract. Less duplication, fewer future regrets.
