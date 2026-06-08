How to train and compare models
===============================

This guide covers the normal operator loop: train a run, inspect the result,
compare it with prior runs, and decide whether it deserves validation and a model
card.

Train the baseline sklearn run
------------------------------

.. code-block:: bash

   uv run bc-mlops train --config configs/train.yaml --output-dir artifacts/runs

Train the baseline PyTorch run
------------------------------

.. code-block:: bash

   uv run bc-mlops train --config configs/train-pytorch.yaml --output-dir artifacts/runs

Train the digits vision CNN example
-----------------------------------

.. code-block:: bash

   uv run bc-mlops train --config configs/train-digits-cnn.yaml --output-dir artifacts/runs

Train Coimbra benchmark configs
-------------------------------

Random forest:

.. code-block:: bash

   uv run bc-mlops train \
     --config configs/train-coimbra-random-forest.yaml \
     --output-dir artifacts/runs

Histogram gradient boosting:

.. code-block:: bash

   uv run bc-mlops train \
     --config configs/train-coimbra-hist-gradient-boosting.yaml \
     --output-dir artifacts/runs

Histogram gradient boosting with stratified k-fold:

.. code-block:: bash

   uv run bc-mlops train \
     --config configs/train-coimbra-hist-gradient-boosting-kfold.yaml \
     --output-dir artifacts/runs

Compare recorded runs
---------------------

Raw registry payload:

.. code-block:: bash

   uv run bc-mlops compare --registry artifacts/registry.json

Compact human-readable summary:

.. code-block:: bash

   uv run bc-mlops compare --registry artifacts/registry.json --summary

Open the dashboard
------------------

Static dashboard:

.. code-block:: bash

   uv run bc-mlops dashboard --registry artifacts/registry.json --run-root artifacts/runs

Interactive deck:

.. code-block:: bash

   uv run bc-mlops dashboard \
     --registry artifacts/registry.json \
     --run-root artifacts/runs \
     --interactive

What to inspect after training
------------------------------

Review these files in ``artifacts/runs/<run-name>/``:

- ``metadata.json`` for dataset, runtime, MLflow, and evaluation metadata
- ``metrics.json`` for scalar evaluation output
- ``config.resolved.yaml`` for the normalized config actually used
- ``fold_metrics.json`` for per-fold metrics and summary stats when k-fold is enabled
- ``feature_importance.csv`` when the backend emits a feature-importance view
- ``MODEL_CARD.md`` after you run ``bc-mlops report``

When to use which view
----------------------

Use ``compare --summary`` when you want a fast terminal ranking.

Use the dashboard when you need:

- richer artifact-health visibility
- run-to-run compare mode
- config browsing
- run designer and model designer workflows
- action shortcuts for validate, report, predict, and retrain

Next steps
----------

After comparing runs, typical follow-up actions are:

- :doc:`validate-and-report`
- :doc:`run-inference`
- :doc:`inspect-mlflow`
