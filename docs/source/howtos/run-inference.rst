How to run offline inference
============================

The ``predict`` command accepts either JSON or CSV payloads and automatically
routes scoring through the saved backend artifact.

Single-record JSON input
------------------------

.. code-block:: bash

   uv run bc-mlops predict \
     --model artifacts/runs/<run-name>/model.joblib \
     --input sample-inputs/sample.json

Batch CSV input
---------------

.. code-block:: bash

   uv run bc-mlops predict \
     --model artifacts/runs/<run-name>/model.pt \
     --input samples/batch.csv

Output shape
------------

The command emits JSON with a ``predictions`` list. Each item contains:

- ``index``
- ``label``
- ``probability``

Troubleshooting
---------------

- Use ``.joblib`` for the scikit-learn backend.
- Use ``.pt`` for the PyTorch backend.
- Ensure feature columns match the training schema stored in the artifact.
