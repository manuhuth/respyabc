"""
respyabc package.

respyabc is an open-source package providing easy implementation of the pyabc package for dynamic discrete choice models created by the respy package.
"""
__version__ = "0.0.0"

import pytest

from respyabc.config import ROOT_DIR


def test(*args, **kwargs):
    """Run basic tests of the package."""
    pytest.main([str(ROOT_DIR), *args], **kwargs)
