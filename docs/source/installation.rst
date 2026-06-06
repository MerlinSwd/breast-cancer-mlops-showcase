Installation
============

Prerequisites
-------------

The repository is built around `uv` and Python 3.11.

- Python 3.11+
- `uv <https://docs.astral.sh/uv/>`_
- Git

Clone and sync
--------------

.. code-block:: bash

   git clone https://github.com/MerlinSwd/breast-cancer-mlops-showcase.git
   cd breast-cancer-mlops-showcase
   uv sync --extra dev --extra docs

This installs:

- the package itself in editable mode
- developer tooling such as pytest, ruff, and build
- Sphinx and related documentation extensions

Verify the environment
----------------------

.. code-block:: bash

   uv run python -m pytest
   uv run ruff check .
   uv run python -m build
   uv run python -m sphinx -b html docs/source docs/_build/html

PyTorch source note
-------------------

The project pins the PyTorch CPU wheel index in ``pyproject.toml`` so that
``uv sync`` remains deterministic in environments without CUDA.

If you want GPU execution, keep the same configuration shape but install a CUDA-
compatible PyTorch distribution and set ``model.device: auto`` or ``cuda`` in
training config.
