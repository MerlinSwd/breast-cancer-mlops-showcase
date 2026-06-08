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
     --model artifacts/runs/<run-name>/model.joblib \
     --input /path/to/batch.csv

Output shape
------------

The command emits JSON with a ``predictions`` list. Each item contains:

- ``index``
- ``label``
- ``probability``

Input requirements
------------------

Your JSON or CSV records must match the feature schema expected by the trained
artifact.

Practical rules:

- do not include the target column
- keep feature column names aligned with the training dataset
- use numeric/categorical encodings compatible with training
- use ``model.joblib`` for scikit-learn backends and ``model.pt`` for PyTorch

Troubleshooting
---------------

If inference fails, check:

- the artifact path suffix matches the backend family
- the input file extension is ``.json`` or ``.csv``
- the input records contain the expected feature columns
- the batch file is a real CSV path, not a wishful docs typo from a previous era
