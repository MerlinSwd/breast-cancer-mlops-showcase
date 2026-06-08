Configuration reference
=======================

The project is configuration-driven. The CLI stays small, while datasets,
evaluation strategy, tracking, and model families are selected in YAML.

Top-level schema
----------------

A training config maps to ``TrainingConfig`` in ``src/bc_mlops_showcase/config.py``.

.. code-block:: yaml

   experiment_name: baseline-logreg
   random_seed: 42
   threshold: 0.5
   split:
     test_size: 0.2
     stratify: true
   evaluation:
     mode: holdout
     folds: 5
   tracking:
     uri: ./mlruns
     experiment_name: bc-mlops-showcase
   dataset:
     kind: sklearn_breast_cancer
   model:
     kind: sklearn_logreg
     device: auto
     params:
       c: 1.0
       max_iter: 500

Fields
------

``experiment_name``
   Free-form run name prefix used for the output directory and the MLflow run name.

``random_seed``
   Shared seed used by supported backends and data splitting.

``threshold``
   Probability threshold used to convert probabilities into positive predictions
   for accuracy, precision, recall, F1, and positive-rate metrics.

``split``
   Holdout split settings.

``evaluation``
   Evaluation strategy. Supported values:

   - ``holdout``
   - ``stratified_k_fold``

``tracking``
   MLflow tracking destination and experiment name.

``dataset``
   Dataset loader selection and loader-specific parameters.

``model``
   Model family, runtime device preference, and model-specific hyperparameters.

Split settings
--------------

.. code-block:: yaml

   split:
     test_size: 0.2
     stratify: true

``test_size``
   Fraction of rows reserved for the test set during holdout evaluation.

``stratify``
   When true, holdout splitting preserves class balance.

Evaluation settings
-------------------

.. code-block:: yaml

   evaluation:
     mode: holdout
     folds: 5

``mode``
   ``holdout`` or ``stratified_k_fold``.

``folds``
   Number of folds when ``mode: stratified_k_fold``. Must be at least 2.

Important limitation
~~~~~~~~~~~~~~~~~~~~

``stratified_k_fold`` currently supports **scikit-learn backends only**. If you
try to use it with ``pytorch_mlp``, training raises a validation error from the
pipeline. The code is being honest here; unlike some dashboards, it has not yet
learned to bluff.

Tracking settings
-----------------

.. code-block:: yaml

   tracking:
     uri: ./mlruns
     experiment_name: bc-mlops-showcase

``uri``
   Tracking destination. Local filesystem paths are resolved into:

   - a SQLite tracking database at ``<uri>/mlflow.db``
   - an artifact root at ``<uri>/artifacts/``

   Full URIs such as ``http://mlflow.example.com`` are passed through unchanged.

``experiment_name``
   MLflow experiment name.

Dataset settings
----------------

Supported dataset kinds
~~~~~~~~~~~~~~~~~~~~~~~

``sklearn_breast_cancer``
   Uses the built-in Wisconsin diagnostic dataset from scikit-learn.

``csv_tabular_binary``
   Loads a CSV file and converts one target label into the positive class.

``sklearn_digits_binary``
   Uses the built-in scikit-learn digits dataset, filtered to digits ``0`` and ``1``
   so the existing binary-classification pipeline can train a compact vision model.

CSV dataset fields
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   dataset:
     kind: csv_tabular_binary
     path: data/breast-cancer-coimbra.csv
     target_column: Classification
     positive_label: 2.0
     drop_columns: []

``path``
   Required for ``csv_tabular_binary``.

``target_column``
   Column to convert into the binary target.

``positive_label``
   Value treated as class ``1``. All other values become class ``0``.

``drop_columns``
   Optional feature columns to remove before training.

See :doc:`datasets` for examples and dataset-specific guidance.

Model settings
--------------

Supported model kinds
~~~~~~~~~~~~~~~~~~~~~

The current supported model families are:

- ``sklearn_logreg``
- ``sklearn_random_forest``
- ``sklearn_hist_gradient_boosting``
- ``pytorch_mlp``
- ``pytorch_cnn``

Device options
~~~~~~~~~~~~~~

Supported device values are:

- ``auto``
- ``cpu``
- ``cuda``

``auto`` resolves to CUDA when available, otherwise CPU.

Default parameter sets
~~~~~~~~~~~~~~~~~~~~~~

``sklearn_logreg``

.. code-block:: yaml

   model:
     kind: sklearn_logreg
     device: auto
     params:
       c: 1.0
       max_iter: 500

``sklearn_random_forest``

.. code-block:: yaml

   model:
     kind: sklearn_random_forest
     device: auto
     params:
       n_estimators: 200
       max_depth: null
       min_samples_leaf: 1

``sklearn_hist_gradient_boosting``

.. code-block:: yaml

   model:
     kind: sklearn_hist_gradient_boosting
     device: auto
     params:
       learning_rate: 0.1
       max_iter: 200
       max_depth: null
       min_samples_leaf: 20

``pytorch_mlp``

.. code-block:: yaml

   model:
     kind: pytorch_mlp
     device: auto
     params:
       hidden_dims: [32, 16]
       epochs: 20
       batch_size: 32
       learning_rate: 0.01
       dropout: 0.1

``pytorch_cnn``

.. code-block:: yaml

   model:
     kind: pytorch_cnn
     device: auto
     params:
       conv_channels: [8, 16]
       kernel_size: 3
       epochs: 8
       batch_size: 32
       learning_rate: 0.005
       hidden_dim: 32

Config examples shipped in the repo
-----------------------------------

The repository includes these ready-to-run config files under ``configs/``:

- ``train.yaml``
- ``train-pytorch.yaml``
- ``train-digits-cnn.yaml``
- ``train-coimbra-random-forest.yaml``
- ``train-coimbra-hist-gradient-boosting.yaml``
- ``train-coimbra-hist-gradient-boosting-kfold.yaml``
- ``quality_gates.yaml``

Validation behavior
-------------------

Config loading rejects:

- unsupported dataset kinds
- unsupported model kinds
- unsupported model devices
- unsupported evaluation modes
- ``evaluation.folds < 2``

The run designer and model designer inside the interactive dashboard reuse these
same rules so the TUI does not invent its own private religion.
