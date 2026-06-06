Usage
=====

CLI overview
------------

The package exposes a single CLI entrypoint named ``bc-mlops``.

.. code-block:: bash

   uv run bc-mlops --help

Available subcommands:

- ``train``: train a configured backend and write run artifacts
- ``compare``: inspect the lightweight experiment registry
- ``validate``: enforce quality gates against a metrics file
- ``predict``: score JSON or CSV payloads against a trained artifact
- ``report``: generate a markdown model card for a run

Train a baseline model
----------------------

.. code-block:: bash

   uv run bc-mlops train --config configs/train.yaml --output-dir artifacts/runs

Train the PyTorch backend
-------------------------

.. code-block:: bash

   uv run bc-mlops train --config configs/train-pytorch.yaml --output-dir artifacts/runs

Compare runs
------------

.. code-block:: bash

   uv run bc-mlops compare --registry artifacts/registry.json

Validate a trained run
----------------------

.. code-block:: bash

   uv run bc-mlops validate \
     --metrics artifacts/runs/<run-name>/metrics.json \
     --gates configs/quality_gates.yaml

Run offline inference
---------------------

.. code-block:: bash

   uv run bc-mlops predict \
     --model artifacts/runs/<run-name>/model.joblib \
     --input sample-inputs/sample.json

   uv run bc-mlops predict \
     --model artifacts/runs/<run-name>/model.pt \
     --input sample-inputs/sample.json

Generate a model card
---------------------

.. code-block:: bash

   uv run bc-mlops report \
     --run-dir artifacts/runs/<run-name> \
     --output artifacts/runs/<run-name>/MODEL_CARD.md

Generated artifacts
-------------------

Each training run creates a timestamped directory containing:

- the serialized model artifact
- ``metrics.json``
- ``metadata.json``
- ``config.resolved.yaml``
- optional ``feature_importance.csv``
- generated ``MODEL_CARD.md`` when requested

MLflow runs are also logged under the configured tracking backend, which defaults
to a local SQLite database plus filesystem artifact store.
