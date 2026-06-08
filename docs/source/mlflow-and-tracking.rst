MLflow and run tracking
=======================

This project stores run information in **three layers**:

- the run directory under ``artifacts/runs/<run-name>/``
- the lightweight registry at ``artifacts/registry.json``
- the MLflow tracking backend configured by ``tracking.uri``

Understanding the difference keeps a lot of confusion from breeding in the dark.

Local default behavior
----------------------

The default tracking config is:

.. code-block:: yaml

   tracking:
     uri: ./mlruns
     experiment_name: bc-mlops-showcase

For local filesystem paths, the tracking helper resolves that into:

- tracking database: ``mlruns/mlflow.db``
- artifact root: ``mlruns/artifacts/``

Why SQLite instead of the legacy file store?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MLflow 3.x deprecated the plain file-store path unless
``MLFLOW_ALLOW_FILE_STORE=true`` is set. Using SQLite avoids that footgun and
keeps local runs queryable.

What gets written where
-----------------------

Run directory
~~~~~~~~~~~~~

The run directory is the most complete source of truth for a specific training
run. It stores artifacts such as:

- the serialized model artifact
- ``metrics.json``
- ``metadata.json``
- ``config.resolved.yaml``
- optional ``fold_metrics.json``
- optional ``feature_importance.csv``
- optional ``MODEL_CARD.md`` after running ``bc-mlops report``

Registry
~~~~~~~~

``artifacts/registry.json`` is a lightweight run index used by ``compare`` and
the dashboard. It is intentionally small and does **not** mirror every artifact
field.

Think of it as a quick leaderboard and lookup table, not a full metadata store.

MLflow
~~~~~~

MLflow receives:

- flattened config parameters
- final scalar metrics
- the full run directory as logged artifacts under ``run_artifacts``
- tags such as ``model.kind`` and ``project``

MLflow is the right place for experiment history. The run directory is the right
place for immediate local inspection. The registry is the fast summary layer.

Remote MLflow servers
---------------------

If ``tracking.uri`` contains ``://``, it is treated as a full URI and passed
through without local path rewriting.

Example:

.. code-block:: yaml

   tracking:
     uri: http://mlflow.example.com
     experiment_name: bc-mlops-showcase

In that case the project still writes local run artifacts under
``artifacts/runs/``, but MLflow logging targets the remote server.

Inspecting a completed run
--------------------------

Start with the local run directory:

- ``artifacts/runs/<run-name>/metadata.json``
- ``artifacts/runs/<run-name>/metrics.json``
- ``artifacts/runs/<run-name>/config.resolved.yaml``
- ``artifacts/runs/<run-name>/fold_metrics.json`` when k-fold evaluation is used

Then inspect the registry or dashboard:

.. code-block:: bash

   uv run bc-mlops compare --registry artifacts/registry.json
   uv run bc-mlops compare --registry artifacts/registry.json --summary
   uv run bc-mlops dashboard --registry artifacts/registry.json --run-root artifacts/runs

Finally, inspect MLflow if you need experiment-wide history or artifact browsing.

What ``metadata.json`` includes
-------------------------------

``metadata.json`` captures run-local metadata such as:

- experiment name and timestamp
- train/test row counts
- dataset metadata
- evaluation mode and fold count
- model kind and runtime information
- config snapshot
- MLflow run identifiers and resolved tracking location

Related pages
-------------

- :doc:`artifacts`
- :doc:`dashboard`
- :doc:`howtos/inspect-mlflow`
