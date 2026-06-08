Architecture
============

This project is a config-driven tabular MLOps system for binary classification.
The central design choice is simple:

- keep orchestration stable
- put dataset and model selection in configuration
- keep operator tooling pointed at the same artifacts and metadata

That choice is doing a lot of work and, unlike some architecture diagrams, is not
merely decorative.

High-level flow
---------------

Training flow:

#. load YAML into ``TrainingConfig``
#. load the selected dataset into pandas objects
#. train the selected backend
#. evaluate with holdout or stratified k-fold
#. write run artifacts to disk
#. log params, metrics, and artifacts to MLflow
#. update the lightweight registry used by compare/dashboard

Core modules
------------

``config.py``
   Loads and validates YAML configuration, resolves defaults, and centralizes
   supported model kinds, device options, and registry-declared parameter
   schemas.

``data.py``
   Loads the built-in Wisconsin dataset, the built-in digits vision dataset, or a
   CSV binary-tabular dataset.

``modeling.py``
   Implements backend-specific training, serialization, and prediction helpers
   for:

   - ``sklearn_logreg``
   - ``sklearn_random_forest``
   - ``sklearn_hist_gradient_boosting``
   - ``pytorch_mlp``
   - ``pytorch_cnn``

``pipeline.py``
   Orchestrates dataset loading, training, evaluation, artifact writing,
   registry updates, and MLflow integration.

``tracking.py``
   Resolves the MLflow backend, starts runs, logs flattened config parameters,
   logs metrics and artifacts, and closes runs.

``inference.py``
   Loads JSON or CSV records and scores them through a saved backend artifact.

``validation.py``
   Applies threshold-based quality gates to ``metrics.json``.

``reporting.py``
   Generates Markdown model cards from completed run artifacts.

``tui.py``
   Implements the static dashboard renderer and the interactive Textual command
   deck.

``designer.py`` and ``model_designer.py``
   Hold the draft-state and validation logic behind the run designer and model
   designer workflows.

System views
------------

Component view
~~~~~~~~~~~~~~

.. mermaid::

   flowchart LR
       CLI[CLI\ncli.py] --> CFG[Configuration\nconfig.py]
       CLI --> PIPE[Training Pipeline\npipeline.py]
       CLI --> INF[Inference\ninference.py]
       CLI --> VAL[Validation\nvalidation.py]
       CLI --> REP[Reporting\nreporting.py]
       CLI --> TUI[Dashboard / TUI\ntui.py]
       PIPE --> DATA[Dataset Loader\ndata.py]
       PIPE --> MODEL[Backends\nmodeling.py]
       PIPE --> TRACK[MLflow Tracking\ntracking.py]
       TUI --> DESIGN[Run + Model Designers\ndesigner.py / model_designer.py]
       TUI --> REG[(registry.json)]
       TUI --> RUNS[(run artifacts)]
       PIPE --> RUNS
       PIPE --> REG
       TRACK --> MLFLOW[(MLflow DB + artifacts)]
       INF --> MODEL
       REP --> RUNS
       VAL --> RUNS

Sequence view for ``train``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   sequenceDiagram
       participant User
       participant CLI as cli.py
       participant Config as config.py
       participant Pipeline as pipeline.py
       participant Data as data.py
       participant Backend as modeling.py
       participant Tracking as tracking.py
       participant Disk as Run Artifacts / Registry

       User->>CLI: bc-mlops train --config ...
       CLI->>Config: load_training_config()
       CLI->>Pipeline: train_and_evaluate(config)
       Pipeline->>Data: load_dataset()
       Pipeline->>Tracking: start_training_run(config)
       Pipeline->>Backend: train_backend(config, X_train, y_train)
       Backend-->>Pipeline: BackendTrainingBundle
       Pipeline->>Disk: write model, metrics, metadata, config snapshot
       Pipeline->>Tracking: finish_training_run(metrics, run_dir)
       Pipeline->>Disk: update registry.json
       Pipeline-->>CLI: TrainingResult
       CLI-->>User: JSON summary

Storage layers
--------------

The system intentionally keeps three storage layers with different purposes.

Run directory
~~~~~~~~~~~~~

``artifacts/runs/<run-name>/`` is the richest local source of truth for one run.
It contains model artifacts, metrics, metadata, resolved config, and optional
extras such as fold summaries and feature importance.

Registry
~~~~~~~~

``artifacts/registry.json`` is a lightweight summary layer used for fast compare
and dashboard views. It is intentionally smaller than ``metadata.json``.

MLflow
~~~~~~

MLflow stores experiment history across runs: config parameters, final metrics,
and a logged copy of the run directory.

Evaluation model
----------------

The pipeline supports two evaluation strategies:

``holdout``
   Train/test split using ``train_test_split``.

``stratified_k_fold``
   Out-of-fold scoring with per-fold metrics written to ``fold_metrics.json``.

Current limitation: ``stratified_k_fold`` is implemented only for
scikit-learn backends. The pipeline enforces this directly.

Dashboard architecture
----------------------

The dashboard has two faces:

- static text rendering for quick inspection
- interactive Textual UI for operations and authoring workflows

The interactive deck has four modes:

- ``runs``
- ``configs``
- ``run-designer``
- ``model-designer``

The run designer owns the full ``TrainingConfig`` draft. The model designer owns
a focused ``ModelConfig`` workbench and applies its result back into the run
Designer draft. That split keeps model tuning ergonomic without turning the TUI
into a YAML confession booth.

Extension strategy
------------------

The future-proofing hinge is a pair of registries in ``config.py``:

- ``DATASET_SPECS`` declares dataset modality and identity
- ``MODEL_SPECS`` declares backend modality, default run naming, supported
  evaluation modes, and editable parameter schemas

That means new datasets and models extend a contract instead of spraying ``if``
statements through the codebase. The pipeline now validates dataset/model and
model/evaluation compatibility early, before training gets the chance to fail in
some deeper and more annoying place.

A new backend should:

- register itself in ``MODEL_SPECS``
- declare its editable params in the model-parameter schema used to derive
  defaults and validation
- train from pandas features and labels
- save a portable artifact through the artifact-loader registry
- support probability prediction for offline inference
- participate in reporting, validation, dashboard, and MLflow flows

A new dataset should:

- register itself in ``DATASET_SPECS``
- expose stable class labels for inference and reporting metadata
- load into pandas features plus binary targets through ``data.py``
- remain compatible with at least one backend/evaluation pairing

The same contract also keeps inference honest: prediction labels now come from
run metadata instead of a hard-coded ``benign/malignant`` fairy tale.

Artifact loading is now registry-driven as well. The training layer records a
small metadata contract alongside each run:

- ``contract.metadata_version``
- ``contract.task``
- ``model.artifact.filename``
- ``model.artifact.format``
- ``model.artifact.loader``
- ``model.artifact.version``

Offline inference resolves the loader from that metadata first and only falls
back to filename suffixes when the contract is absent. So future model formats
can arrive without teaching every downstream surface to guess from ``.joblib``
versus ``.pt`` like it's reading tea leaves.

The model designer now consumes those same schemas rather than hard-coded
per-backend fields. Draft state stores generic parameter text values, and the
Textual form shows only the active family's registry-declared controls. That
reduces config/TUI drift when new backends appear.

Additional areas worth future-proofing next:

- move metric/quality-gate definitions toward task-aware registries if the project expands past binary classification

That gives you swappable models with stable operational plumbing instead of a
special-case festival.
