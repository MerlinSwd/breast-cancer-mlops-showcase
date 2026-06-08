Datasets
========

The project currently supports three dataset loaders. All three produce pandas
features/target objects and feed the same training pipeline.

Supported dataset kinds
-----------------------

``sklearn_breast_cancer``
   Uses the built-in Wisconsin diagnostic dataset shipped with scikit-learn.

``csv_tabular_binary``
   Loads a CSV file, converts a configured label into the positive class, and
   uses the remaining columns as features.

``sklearn_digits_binary``
   Uses the built-in handwritten-digits dataset from scikit-learn, restricted to
   digits ``0`` and ``1`` so the project can train a binary CNN without changing
   the surrounding training and evaluation contracts.

Built-in Wisconsin dataset
--------------------------

This is the default dataset used by ``configs/train.yaml`` and
``configs/train-pytorch.yaml``.

Characteristics:

- loaded from ``sklearn.datasets.load_breast_cancer(as_frame=True)``
- binary target already present in the source dataset
- no file path required
- useful as the fastest smoke-test dataset for local development and CI

Minimal config:

.. code-block:: yaml

   dataset:
     kind: sklearn_breast_cancer

Built-in digits vision dataset
------------------------------

``configs/train-digits-cnn.yaml`` uses the built-in scikit-learn digits dataset,
filtered to digits ``0`` and ``1``. The loader flattens each ``8x8`` grayscale
image into ``64`` numeric features so the existing artifact and inference
contracts stay stable, while the CNN backend reshapes those features back into an
image tensor during training and prediction.

Characteristics:

- loaded from ``sklearn.datasets.load_digits()``
- filtered to a binary task without needing an external image directory
- fast enough for unit tests and local smoke runs
- designed to pair with ``model.kind: pytorch_cnn``

Minimal config:

.. code-block:: yaml

   dataset:
     kind: sklearn_digits_binary

CSV binary-tabular dataset
--------------------------

This loader powers the Coimbra benchmark configs.

Required fields:

.. code-block:: yaml

   dataset:
     kind: csv_tabular_binary
     path: data/breast-cancer-coimbra.csv
     target_column: Classification
     positive_label: 2.0
     drop_columns: []

How the loader works
~~~~~~~~~~~~~~~~~~~~

For ``csv_tabular_binary``:

#. read the CSV with pandas
#. verify that ``target_column`` exists
#. convert ``frame[target_column] == positive_label`` into class ``1``
#. convert every other label value into class ``0``
#. drop the target column and any configured ``drop_columns``
#. treat all remaining columns as model features

Practical implications
~~~~~~~~~~~~~~~~~~~~~~

- ``positive_label`` is not cosmetic; it defines what probability means during
  evaluation and inference.
- The model artifact expects the same feature column names at inference time.
- ``drop_columns`` only removes columns that actually exist in the CSV.

Coimbra benchmark guidance
--------------------------

The repository includes multiple configs for the Coimbra dataset because it is a
smaller and trickier benchmark than the built-in Wisconsin dataset.

Available examples:

- ``configs/train-coimbra-random-forest.yaml``
- ``configs/train-coimbra-hist-gradient-boosting.yaml``
- ``configs/train-coimbra-hist-gradient-boosting-kfold.yaml``

Why stratified k-fold is useful here
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Small tabular datasets can produce noisy holdout metrics when a single split gets
lucky or cursed. ``evaluation.mode: stratified_k_fold`` evaluates the model from
out-of-fold predictions and also writes ``fold_metrics.json`` so you can inspect
mean/std stability across folds.

Reminder: stratified k-fold currently supports **scikit-learn backends only**.

Inference schema expectations
-----------------------------

Offline inference loads JSON or CSV into a pandas frame and passes it to the
saved model artifact.

That means your input records must:

- contain the feature columns expected by the trained artifact
- use compatible numeric/categorical encodings with the training data
- omit the target column

If your CSV training config used ``drop_columns``, those dropped columns should
not reappear as inference features unless your artifact was trained with them.

Related pages
-------------

- :doc:`configuration`
- :doc:`artifacts`
- :doc:`howtos/train-models`
- :doc:`howtos/run-inference`
