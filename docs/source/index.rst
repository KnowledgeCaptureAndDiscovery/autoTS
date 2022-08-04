.. autoTS documentation master file, created by
   sphinx-quickstart on Thu Jun  2 12:49:49 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

autoTS: automated time series analysis
======================================

autoTS is an automated system for time series analysis. autoTS is conceived around three key ideas:

* To capture expert knowledge as semantic workflow templates that represent common multi-step analysis strategies and constraints,

* To automatically characterize datasets to extract properties that will affect the choice of analytic methods (e.g., the data has missing values) to instantiate those workflow templates,

* To curate a catalog of primitive pre-processing and analysis functions for time-series data that map to workflow steps and are annotated with preconditions (e.g., evenly-spaced time series is required), performance (e.g., bias in the low-frequency domain), and other constraints that will be used to generate solutions.


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
