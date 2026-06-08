Dashboard and interactive command deck
======================================

The ``dashboard`` command has two modes:

- a static terminal dashboard
- an interactive Textual command deck

Both read the lightweight run registry plus run-artifact directories.

Launch commands
---------------

Static dashboard:

.. code-block:: bash

   uv run bc-mlops dashboard \
     --registry artifacts/registry.json \
     --run-root artifacts/runs

Interactive deck:

.. code-block:: bash

   uv run bc-mlops dashboard \
     --registry artifacts/registry.json \
     --run-root artifacts/runs \
     --interactive

Static dashboard
----------------

The static dashboard is a fast read-only summary intended for terminal use and CI
artifacts. It highlights:

- the current champion run
- a leaderboard across tracked runs
- artifact-health checks for each run directory
- registry-versus-disk drift such as missing run directories or stale entries
- next-step operator hints such as validation or model-card generation

Use this mode when you want a quick operational read without opening the full
interactive UI.

Interactive command deck
------------------------

The interactive deck is the operator console. It combines run browsing,
configuration browsing, comparison, and authoring workflows in one TUI.

Main capabilities
~~~~~~~~~~~~~~~~~

- live filtering by run name or model kind
- keyboard navigation through tracked runs
- richer run details including artifact paths and MLflow identifiers
- artifact drill-down for generated files
- operator actions for ``validate``, ``report``, ``predict``, and ``retrain``
- run-to-run compare mode
- config browser mode
- run designer for full training-config drafting
- model designer for guided ``model:`` editing
- task-status panel for action results and errors

Modes and lanes
---------------

The interactive deck uses four top-level modes:

``runs``
   Browse tracked runs and inspect their artifacts.

``configs``
   Browse saved training YAML files under ``configs/``.

``run-designer``
   Draft, validate, preview, save, and launch a full training configuration.

``model-designer``
   Tune the ``model`` section visually, preview normalized model YAML, and apply
   the result back into the run designer.

Run designer workflow
---------------------

The run designer owns the full ``TrainingConfig`` draft.

Typical flow:

#. open the run designer
#. start from defaults or clone a saved config
#. edit dataset, evaluation, tracking, and model fields
#. preview normalized YAML
#. validate the draft
#. save it under ``configs/``
#. launch training directly from the TUI

Model designer workflow
-----------------------

The model designer is a focused workbench for model-family tuning.

Typical flow:

#. open the model designer from the toolbar or with ``b``
#. choose a supported model family
#. edit family-specific hyperparameters with guided controls
#. preview normalized ``model:`` YAML
#. validate the draft
#. apply the result back into the run designer

Supported model families match the codebase:

- ``sklearn_logreg``
- ``sklearn_random_forest``
- ``sklearn_hist_gradient_boosting``
- ``pytorch_mlp``

Keyboard shortcuts
------------------

Common deck shortcuts:

- ``tab``: cycle modes
- ``/``: focus the filter box
- ``enter``: cycle details and artifact views
- ``a``: action catalog
- ``c``: set compare anchor
- ``?``: help
- ``r``: reload
- ``q``: quit

Designer shortcuts:

- ``n``: open run designer
- ``b``: open model designer

What the dashboard reads
------------------------

The dashboard is not a second training system. It reads project outputs:

- ``artifacts/registry.json`` for the lightweight run index
- ``artifacts/runs/<run-name>/`` for real run artifacts
- saved YAMLs under ``configs/`` for config browsing and cloning

See :doc:`artifacts` for the exact file inventory and :doc:`mlflow-and-tracking`
for the relationship between the run directory, registry, and MLflow.
