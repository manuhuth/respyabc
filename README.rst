.. |logo| image:: https://raw.githubusercontent.com/OpenSourceEconomics/ose-corporate-design/master/logos/OSE_logo_no_type_RGB.svg
   :height: 25px

respyabc
==============
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

For more information on the package check out respyabc at its `online documentation <https://respyabc.readthedocs.io/en/latest/>`_.

Installing
==============
With ``conda`` and ``pip`` available on your path, installing
``respyabc`` is as simple as typing

.. code-block:: bash

    $ pip install pyabc
    $ conda config --add channels conda-forge
    $ conda install -c opensourceeconomics respy
    $ conda install -c manuhuth respyabc
    
Repository structure
=======================
The root directory of this repository contains two main folders ``respyabc`` and ``docs``. 

- ``respyabc``: The folder ``respyabc`` contains all modules and tests that are written for the package. An overview of the models is given at respyabc's `API reference <https://respyabc.readthedocs.io/en/latest/api.html>`_. The folder ``respyabc/tests`` contains all tests that are conducted in order to ensure the functionality of the package. 

- ``docs``: The folder ``docs`` contains all files that describe the used modules and are used to build the `documentation <https://respyabc.readthedocs.io/en/latest/>`_. Example notebooks can be found in ``docs/source/tutorials``.

Example project
===================================
You can find an example project that showcases how respyabc can be usd in an actual research paper in this `repository <https://github.com/manuhuth/respyabc_application>`_.
