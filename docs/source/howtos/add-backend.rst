How to add a new backend
========================

This repository is designed so model swaps happen in configuration rather than
through CLI churn. Adding a new backend means extending a shared contract, not
smuggling special cases through every module.

1. Extend configuration support
-------------------------------

Update ``src/bc_mlops_showcase/config.py``:

- register the backend in ``MODEL_SPECS`` with label, modality, default experiment name, and supported evaluation modes
- add default hyperparameters in ``DEFAULT_MODEL_PARAMS``
- let ``_resolve_model_config()`` merge and normalize the backend params

If the backend supports device selection, keep using the existing device contract:

- ``auto``
- ``cpu``
- ``cuda``

2. Implement training logic
---------------------------

Add backend implementation in ``src/bc_mlops_showcase/modeling.py`` that can:

- train from pandas features and labels
- return a ``BackendTrainingBundle``
- persist a portable artifact
- predict positive-class probabilities during inference
- optionally emit a feature-importance view

3. Register the backend
-----------------------

Update ``train_backend()`` so that the new ``model.kind`` maps to your trainer.

If the artifact format is new, also update the inference loading path so
``predict`` can score saved models.

4. Check dataset compatibility
------------------------------

Review ``DATASET_SPECS`` and the compatibility validation in ``config.py``.
Backends should declare the modality they expect, and unsupported evaluation
modes should fail during config loading rather than halfway through a run.

5. Preserve prediction labels
-----------------------------

If the backend changes dataset semantics or artifact metadata, make sure the run
still writes stable dataset labels into ``metadata.json`` so offline inference
returns domain-correct labels.

6. Add config examples
----------------------

Create a YAML config under ``configs/`` showing the new backend’s hyperparameters
and, if relevant, a realistic dataset choice.

7. Update operator surfaces
---------------------------

Because the dashboard contains authoring workflows now, also review:

- ``src/bc_mlops_showcase/designer.py``
- ``src/bc_mlops_showcase/model_designer.py``
- ``src/bc_mlops_showcase/tui.py``

At minimum, make sure the run designer and model designer do not drift from the
new backend’s config contract.

8. Add tests
------------

Cover at least:

- config loading and validation
- backend training bundle behavior
- artifact save/load path
- offline inference
- reporting compatibility if runtime metadata changes
- designer/model-designer validation if new params are exposed there

9. Verify the full lifecycle
----------------------------

Run the same contract as existing backends:

.. code-block:: bash

   uv run bc-mlops train --config configs/<your-backend>.yaml --output-dir artifacts/runs
   uv run bc-mlops validate --metrics artifacts/runs/<run-name>/metrics.json --gates configs/quality_gates.yaml
   uv run bc-mlops predict --model artifacts/runs/<run-name>/<artifact> --input sample-inputs/sample.json
   uv run bc-mlops report --run-dir artifacts/runs/<run-name> --output artifacts/runs/<run-name>/MODEL_CARD.md
   uv run ruff check .
   uv run python -m pytest tests/ -q

10. Update docs
---------------

Before merging, update:

- :doc:`../configuration`
- :doc:`../architecture`
- :doc:`../dashboard` if designer UX changes
- README quick capability list if the new backend is user-facing

The goal is consistency: backend-specific implementation, shared pipeline
contract, and docs that do not lie with confidence.
