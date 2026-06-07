How to train and compare models
===============================

Train the scikit-learn baseline
-------------------------------

.. code-block:: bash

   uv run bc-mlops train --config configs/train.yaml --output-dir artifacts/runs

Train the PyTorch baseline
--------------------------

.. code-block:: bash

   uv run bc-mlops train --config configs/train-pytorch.yaml --output-dir artifacts/runs

Train the Coimbra random-forest benchmark
-----------------------------------------

.. code-block:: bash

   uv run bc-mlops train --config configs/train-coimbra-random-forest.yaml --output-dir artifacts/runs

Train the Coimbra histogram-gradient-boosting benchmark
-------------------------------------------------------

.. code-block:: bash

   uv run bc-mlops train --config configs/train-coimbra-hist-gradient-boosting.yaml --output-dir artifacts/runs

Train the Coimbra histogram-gradient-boosting benchmark with stratified k-fold
------------------------------------------------------------------------------

.. code-block:: bash

   uv run bc-mlops train --config configs/train-coimbra-hist-gradient-boosting-kfold.yaml --output-dir artifacts/runs

Compare all recorded runs
-------------------------

.. code-block:: bash

   uv run bc-mlops compare --registry artifacts/registry.json

Validate the latest run
-----------------------

.. code-block:: bash

   latest_run=$(find artifacts/runs -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
   uv run bc-mlops validate --metrics "$latest_run/metrics.json" --gates configs/quality_gates.yaml

What to inspect after training
------------------------------

Review these files in the run directory:

- ``metadata.json`` for backend/runtime/MLflow metadata and whether the run used holdout or stratified k-fold evaluation
- ``metrics.json`` for evaluation output
- ``fold_metrics.json`` for per-fold metrics plus mean/std summaries when ``evaluation.mode: stratified_k_fold``
- ``config.resolved.yaml`` for the fully resolved training config, including ``evaluation.mode``
- ``feature_importance.csv`` when the backend emits feature importance
- ``MODEL_CARD.md`` after running the dashboard-suggested ``bc-mlops report`` command; k-fold runs include fold-summary snippets there too

The dashboard and interactive command deck now surface ready-to-run operator commands for
``bc-mlops validate`` and ``bc-mlops report`` so the next verification step is visible next to
artifact health, champion selection, and the evaluation mode that produced a highlighted run.
For stratified k-fold runs they also surface cross-validation stability directly from
``fold_metrics.json`` so operators can compare per-run ``F1 σ`` values and dossier-level
mean/std summaries without opening the raw JSON by hand.
