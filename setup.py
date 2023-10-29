# Copyright (c) 2023, Alessandro Abate, Alec Edwards, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="Fossil",
    version="2.0.0",
    author="Alec Edwards",
    author_email="alec.edwards@cs.ox.ac.uk",
    description="Fossil proves properties of dynamical systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/yourname/example_pkg",
    packages=["fossil"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3-Clause",
        "Operating System :: Ubuntu 22.04",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "sympy",
        "matplotlib",
        "z3-solver",
        "cvc5",
        "dreal",
        "scipy",
        "tqdm",
        "pyyaml",
        "pyparsing",
    ],
    entry_points={"console_scripts": ["fossil = fossil.main:_cli_entry"]},
)
