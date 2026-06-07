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

- ``metadata.json`` for backend/runtime/MLflow metadata
- ``metrics.json`` for evaluation output
- ``config.resolved.yaml`` for the fully resolved training config
- ``feature_importance.csv`` when the backend emits feature importance
