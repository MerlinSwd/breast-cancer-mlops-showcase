How to use stratified k-fold evaluation
=======================================

Use stratified k-fold when a small dataset makes single holdout splits too noisy.
The repository includes a ready-made Coimbra example.

Use the shipped config
----------------------

.. code-block:: bash

   uv run bc-mlops train \
     --config configs/train-coimbra-hist-gradient-boosting-kfold.yaml \
     --output-dir artifacts/runs

Enable it in your own config
----------------------------

Set:

.. code-block:: yaml

   evaluation:
     mode: stratified_k_fold
     folds: 5

What changes
------------

With stratified k-fold evaluation, the pipeline:

- trains one final backend artifact on the full dataset
- computes out-of-fold probabilities for metrics
- writes ``fold_metrics.json`` with per-fold metrics and summary stats
- records the evaluation mode and fold count in ``metadata.json``

Important limitation
--------------------

``stratified_k_fold`` currently supports **scikit-learn backends only**.
Using it with ``pytorch_mlp`` raises a pipeline error.

What to inspect
---------------

After the run finishes, inspect:

- ``artifacts/runs/<run-name>/fold_metrics.json``
- ``artifacts/runs/<run-name>/metadata.json``
- ``artifacts/runs/<run-name>/MODEL_CARD.md`` after generating a report

The model card includes a compact cross-validation summary when
``fold_metrics.json`` exists.
