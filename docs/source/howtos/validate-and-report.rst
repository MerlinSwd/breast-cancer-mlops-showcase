How to validate a run and generate a model card
===============================================

After training, validate the metrics against quality gates and then generate a
Markdown model card for the run.

Validate metrics
----------------

.. code-block:: bash

   uv run bc-mlops validate \
     --metrics artifacts/runs/<run-name>/metrics.json \
     --gates configs/quality_gates.yaml

The command returns:

- exit code ``0`` when all checks pass
- exit code ``1`` when any threshold fails

The response payload includes a ``checks`` list with metric name, actual value,
threshold, and pass/fail status.

Generate the model card
-----------------------

.. code-block:: bash

   uv run bc-mlops report \
     --run-dir artifacts/runs/<run-name> \
     --output artifacts/runs/<run-name>/MODEL_CARD.md

What the model card includes
----------------------------

The generated card summarizes:

- experiment name and timestamp
- train/test row counts
- dataset kind and target column
- model kind and runtime
- evaluation mode and fold count
- scalar metrics
- MLflow experiment and run identifiers
- positive-class threshold note
- cross-validation summary when ``fold_metrics.json`` exists

Operator tip
------------

The interactive dashboard exposes ``validate`` and ``report`` as in-TUI actions,
so you can trigger both workflows without leaving the command deck.
