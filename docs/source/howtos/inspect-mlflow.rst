How to inspect MLflow tracking
==============================

The local default tracking setup resolves ``tracking.uri: ./mlruns`` into two
concrete storage locations:

- tracking database: ``mlruns/mlflow.db``
- artifact root: ``mlruns/artifacts/``

Why SQLite instead of plain file store?
---------------------------------------

MLflow 3.x deprecated the legacy file-store-only tracking path unless
``MLFLOW_ALLOW_FILE_STORE=true`` is set. Using SQLite avoids that footgun and
keeps local runs queryable.

Inspect a completed run locally
-------------------------------

After training, inspect:

- ``artifacts/runs/<run-name>/metadata.json``
- ``artifacts/runs/<run-name>/metrics.json``
- ``artifacts/runs/<run-name>/config.resolved.yaml``
- the MLflow artifact copy under ``mlruns/artifacts/``

Understand the split of responsibility
--------------------------------------

- run directory: richest per-run local artifact view
- ``registry.json``: lightweight compare/dashboard index
- MLflow: experiment history, flattened params, scalar metrics, logged artifacts

Use a custom backend
--------------------

If you want to point at a remote MLflow server, set ``tracking.uri`` to a full
URI such as ``http://mlflow.example.com``. The tracking helper preserves full
URIs and only rewrites local filesystem paths.

What the training code logs
---------------------------

During training, the project logs:

- flattened config parameters
- final scalar metrics
- the run directory contents under MLflow artifacts
- tags including ``model.kind`` and ``project``

See also
--------

- :doc:`../mlflow-and-tracking`
- :doc:`../artifacts`
