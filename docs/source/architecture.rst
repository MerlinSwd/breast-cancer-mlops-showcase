Architecture
============

The repository is structured as a backend-driven MLOps pipeline for tabular
binary classification. The guiding design choice is simple: keep orchestration
stable and allow model families to change behind a backend contract.

Core components
---------------

``config.py``
   Loads YAML training configuration and resolves backend defaults.

``data.py``
   Loads the scikit-learn breast cancer dataset into pandas structures.

``modeling.py``
   Implements backend-specific training and artifact loading for
   ``sklearn_logreg`` and ``pytorch_mlp``.

``pipeline.py``
   Orchestrates training, evaluation, artifact writing, registry updates, and
   MLflow integration.

``tracking.py``
   Bootstraps MLflow and manages run lifecycle.

``inference.py``
   Loads saved artifacts and scores JSON/CSV payloads.

``validation.py``
   Checks metric outputs against configured quality gates.

``reporting.py``
   Generates model cards for completed runs.

How things interact
-------------------

The CLI is intentionally thin. It parses user intent and hands off to the
appropriate module. The training path is:

1. Load YAML configuration.
2. Load the dataset.
3. Resolve the requested backend.
4. Train and evaluate the backend.
5. Persist artifacts and metadata.
6. Log the run to MLflow.
7. Update the lightweight experiment registry.

UML diagrams
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
       PIPE --> DATA[Dataset Loader\ndata.py]
       PIPE --> MODEL[Backend Registry\nmodeling.py]
       PIPE --> TRACK[MLflow Tracking\ntracking.py]
       PIPE --> RUNS[(Run Artifacts)]
       TRACK --> MLFLOW[(MLflow DB + Artifacts)]
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
       participant Disk as Artifacts/Registry

       User->>CLI: bc-mlops train --config ...
       CLI->>Config: load_training_config()
       CLI->>Pipeline: train_and_evaluate(config)
       Pipeline->>Data: load_dataset()
       Pipeline->>Tracking: start_training_run(config)
       Pipeline->>Backend: train_backend(config, X_train, y_train)
       Backend-->>Pipeline: BackendTrainingBundle
       Pipeline->>Disk: write metrics, metadata, config snapshot, model
       Pipeline->>Tracking: finish_training_run(metrics, run_dir)
       Pipeline->>Disk: update registry.json
       Pipeline-->>CLI: TrainingResult
       CLI-->>User: JSON summary

Class view of configuration and training result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   classDiagram
       class TrainingConfig {
         +str experiment_name
         +int random_seed
         +float threshold
         +SplitConfig split
         +TrackingConfig tracking
         +ModelConfig model
       }
       class SplitConfig {
         +float test_size
         +bool stratify
       }
       class TrackingConfig {
         +str uri
         +str experiment_name
       }
       class ModelConfig {
         +str kind
         +str device
         +dict params
       }
       class TrainingResult {
         +Path run_dir
         +Path model_path
         +Path metrics_path
         +Path metadata_path
       }
       TrainingConfig --> SplitConfig
       TrainingConfig --> TrackingConfig
       TrainingConfig --> ModelConfig

Extension strategy
------------------

The backend abstraction exists so new model families can be added without
rewriting the CLI or pipeline contract. A new backend should:

- declare config defaults
- train from pandas inputs
- save a portable artifact
- implement probability prediction
- participate in the same metrics, reporting, and MLflow flow

That gives you swappable models with stable operational plumbing. Rare moment of
engineering restraint; cherish it.
