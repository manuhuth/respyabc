.. respyabc documentation master file, created by
   sphinx-quickstart on Sun Mar 21 15:34:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to respyabc's documentation!
====================================
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT

.. image:: https://github.com/manuhuth/respyabc/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/manuhuth/respyabc/actions
   
.. image:: https://codecov.io/gh/manuhuth/respyabc/branch/main/graph/badge.svg?token=KvBaFo3XY3
    :target: https://codecov.io/gh/manuhuth/respyabc
    
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. image:: https://anaconda.org/manuhuth/respyabc/badges/version.svg   
    :target: https://anaconda.org/manuhuth/respyabc

respyabc is a package that uses a likelihood-free inference framework to infer parameters from dynamic discrete choice models. Inference is conducted using Approximate Bayesian Computing and a Sequential Monte-Carlo algorithm via `pyABC <https://pyabc.readthedocs.io/en/latest/>`_. Models must be simulated via `respy <https://respy.readthedocs.io/en/latest/>`_. Currently, only the model of Keane and Wolpin `(1994) <https://www.jstor.org/stable/2109768?seq=1/>`_ is implemented. The extension to further models is the next step of the development phase.

The package has been built and is maintained by Manuel Huth within the scope of the courses Effective Programming Practices for Economists and Scientific Computing, which are taught within the University of Bonn's Master in Economics.

With ``conda`` available on your path, installing
``respyabc`` is as simple as typing

.. code-block:: bash

    $ pip install pyabc
    $ conda config --add channels conda-forge
    $ conda install -c opensourceeconomics respy
    $ conda install -c manuhuth respyabc

Examples
================
Check out the tutorials on this website or you can find an example project that showcases how respyabc can be used in an actual research paper in this `repository <https://github.com/manuhuth/respyabc_application>`_.

.. toctree::
   :maxdepth: 1
   :caption: OSE grading
   
   tutorials/ose_grading.ipynb
   tutorials/ose_grading_fast_run.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   
   tutorials/one_parameter.ipynb
   tutorials/two_parameter.ipynb
   tutorials/model_selection.ipynb
   
.. toctree::
   :maxdepth: 2
   :caption: Api Reference
   
   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
