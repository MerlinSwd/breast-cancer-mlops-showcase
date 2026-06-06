How to inspect MLflow tracking
==============================

The default local tracking setup resolves ``tracking.uri: ./mlruns`` into two
concrete storage locations:

- tracking database: ``mlruns/mlflow.db``
- artifact root: ``mlruns/artifacts/``

Why SQLite instead of plain file store?
---------------------------------------

MLflow 3.x deprecated the legacy file-store-only tracking path unless
``MLFLOW_ALLOW_FILE_STORE=true`` is set. Using SQLite avoids that footgun and
keeps runs queryable.

Inspect generated metadata
--------------------------

After training, inspect:

- ``artifacts/runs/<run-name>/metadata.json``
- ``artifacts/runs/<run-name>/metrics.json``
- the MLflow artifact copy under ``mlruns/artifacts/``

Use a custom backend
--------------------

If you want to point at a remote MLflow server, set ``tracking.uri`` to a full
URI such as ``http://mlflow.example.com``. The tracking helper preserves full
URIs and only rewrites local filesystem paths.
