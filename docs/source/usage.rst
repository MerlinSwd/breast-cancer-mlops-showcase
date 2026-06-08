Usage
=====

The package exposes a single CLI entrypoint named ``bc-mlops``.

.. code-block:: bash

   uv run bc-mlops --help

Available subcommands
---------------------

- ``train``: train a configured backend and write run artifacts
- ``compare``: inspect the lightweight experiment registry
- ``dashboard``: render the static dashboard or interactive command deck
- ``validate``: enforce quality gates against a metrics file
- ``predict``: score JSON or CSV payloads against a trained artifact
- ``report``: generate a markdown model card for a completed run

Typical workflow
----------------

1. pick a config under ``configs/``
2. train a run with ``bc-mlops train``
3. inspect the registry or dashboard
4. validate the run against quality gates
5. generate a model card
6. run offline inference against the trained artifact

Train a run
-----------

Baseline sklearn logistic regression:

.. code-block:: bash

   uv run bc-mlops train --config configs/train.yaml --output-dir artifacts/runs

Baseline PyTorch MLP:

.. code-block:: bash

   uv run bc-mlops train --config configs/train-pytorch.yaml --output-dir artifacts/runs

Digits vision CNN example:

.. code-block:: bash

   uv run bc-mlops train --config configs/train-digits-cnn.yaml --output-dir artifacts/runs

Coimbra random forest benchmark:

.. code-block:: bash

   uv run bc-mlops train \
     --config configs/train-coimbra-random-forest.yaml \
     --output-dir artifacts/runs

Coimbra histogram gradient boosting benchmark:

.. code-block:: bash

   uv run bc-mlops train \
     --config configs/train-coimbra-hist-gradient-boosting.yaml \
     --output-dir artifacts/runs

Coimbra k-fold benchmark:

.. code-block:: bash

   uv run bc-mlops train \
     --config configs/train-coimbra-hist-gradient-boosting-kfold.yaml \
     --output-dir artifacts/runs

Important evaluation note
~~~~~~~~~~~~~~~~~~~~~~~~~

``evaluation.mode: stratified_k_fold`` is currently supported only for
scikit-learn model kinds.

Compare runs
------------

Raw registry output:

.. code-block:: bash

   uv run bc-mlops compare --registry artifacts/registry.json

Human-readable summary:

.. code-block:: bash

   uv run bc-mlops compare --registry artifacts/registry.json --summary

Open the dashboard
------------------

Static terminal dashboard:

.. code-block:: bash

   uv run bc-mlops dashboard \
     --registry artifacts/registry.json \
     --run-root artifacts/runs

Interactive command deck:

.. code-block:: bash

   uv run bc-mlops dashboard \
     --registry artifacts/registry.json \
     --run-root artifacts/runs \
     --interactive

See :doc:`dashboard` for the full TUI workflow, including the run designer and
model designer.

Validate a run
--------------

.. code-block:: bash

   uv run bc-mlops validate \
     --metrics artifacts/runs/<run-name>/metrics.json \
     --gates configs/quality_gates.yaml

The command exits with code ``0`` when all gates pass and ``1`` otherwise.

Generate a model card
---------------------

.. code-block:: bash

   uv run bc-mlops report \
     --run-dir artifacts/runs/<run-name> \
     --output artifacts/runs/<run-name>/MODEL_CARD.md

Run offline inference
---------------------

Single-record JSON input:

.. code-block:: bash

   uv run bc-mlops predict \
     --model artifacts/runs/<run-name>/model.joblib \
     --input sample-inputs/sample.json

CSV batch input:

.. code-block:: bash

   uv run bc-mlops predict \
     --model artifacts/runs/<run-name>/model.joblib \
     --input /path/to/batch.csv

The command prints JSON with a ``predictions`` list containing ``index``,
``label``, and ``probability`` fields.

Where to look after training
----------------------------

Inspect these outputs under ``artifacts/runs/<run-name>/``:

- ``metrics.json``
- ``metadata.json``
- ``config.resolved.yaml``
- ``fold_metrics.json`` for k-fold evaluation
- ``feature_importance.csv`` when emitted by the backend
- ``MODEL_CARD.md`` after running ``bc-mlops report``

For broader context across runs, use:

- ``artifacts/registry.json``
- ``bc-mlops compare``
- ``bc-mlops dashboard``
- MLflow tracking under the configured backend

Related pages
-------------

- :doc:`configuration`
- :doc:`datasets`
- :doc:`dashboard`
- :doc:`artifacts`
- :doc:`mlflow-and-tracking`
- :doc:`howtos/index`
