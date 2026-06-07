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
- ``dashboard``: render a branded terminal dashboard for run metrics and artifact health
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

Train the harder Coimbra benchmark with random forest
-----------------------------------------------------

.. code-block:: bash

   uv run bc-mlops train --config configs/train-coimbra-random-forest.yaml --output-dir artifacts/runs

Train the harder Coimbra benchmark with histogram gradient boosting
-------------------------------------------------------------------

.. code-block:: bash

   uv run bc-mlops train --config configs/train-coimbra-hist-gradient-boosting.yaml --output-dir artifacts/runs

Train the harder Coimbra benchmark with stratified k-fold evaluation
--------------------------------------------------------------------

.. code-block:: bash

   uv run bc-mlops train --config configs/train-coimbra-hist-gradient-boosting-kfold.yaml --output-dir artifacts/runs

Compare runs
------------

.. code-block:: bash

   uv run bc-mlops compare --registry artifacts/registry.json

Open the terminal dashboard
---------------------------

.. code-block:: bash

   uv run bc-mlops dashboard \
     --registry artifacts/registry.json \
     --run-root artifacts/runs

Open the interactive command deck
---------------------------------

.. code-block:: bash

   uv run bc-mlops dashboard \
     --registry artifacts/registry.json \
     --run-root artifacts/runs \
     --interactive

The dashboard highlights the current champion run, prints a sorted leaderboard,
checks whether each run directory still contains the expected artifacts, surfaces
registry-versus-disk drift such as orphan run directories or stale registry
entries, and points the operator at the next command worth running.

The interactive deck adds live filtering by run name or model kind, keyboard
navigation through the run list, an overview pane for deck health and sort
state, a richer run dossier with timestamp, train/test rows, runtime, dataset,
MLflow identifiers, and artifact-path health, a detail pane with metric deltas
versus the champion, an unhealthy-only filter with ``h``, sort cycling with
``s``, filter focus with ``/``, reload support with ``r``, and quit with ``q``.

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

For small datasets such as Coimbra, set ``evaluation.mode: stratified_k_fold`` in
the training config to score the run from out-of-fold predictions while still
writing one final fitted sklearn artifact for downstream inference.
