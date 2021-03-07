from setuptools import find_packages
from setuptools import setup

p = find_packages()

setup(
    name="respyabc",
    version="0.0.0",
    description=(
        """Implementation of the pyabc package for discrete dynamic choice
           models produced by the respy package."""
    ),
    license="MIT",
    url="https://github.com/manuhuth/respyabc",
    author="manuhuth_try",
    author_email="manuel.huth@yahoo.com",
    packages=p,
    zip_safe=False,
    package_data={"utilities": []},
    include_package_data=True,
)
