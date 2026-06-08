Breast Cancer MLOps Showcase
============================

This project demonstrates a small but real tabular MLOps stack for binary breast
cancer classification. The codebase centers on one idea: keep orchestration
stable, and make datasets, model families, and operator workflows configurable.

Use this documentation by job-to-be-done instead of wandering the repo like a
lost intern with grep.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   usage

.. toctree::
   :maxdepth: 2
   :caption: Guides

   datasets
   configuration
   dashboard
   mlflow-and-tracking
   artifacts
   howtos/index

.. toctree::
   :maxdepth: 2
   :caption: Explanation

   architecture

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

Suggested reading paths
-----------------------

If you are new to the project:

#. :doc:`installation`
#. :doc:`usage`
#. :doc:`dashboard`

If you want to operate trained runs:

#. :doc:`usage`
#. :doc:`artifacts`
#. :doc:`mlflow-and-tracking`
#. :doc:`howtos/index`

If you want to extend the system:

#. :doc:`configuration`
#. :doc:`datasets`
#. :doc:`architecture`
#. :doc:`howtos/add-backend`
