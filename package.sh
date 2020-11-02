#!/usr/bin/env bash

find . -name __pycache__ -exec rm -rf {} \;
find . -name '*.pyc' -delete
zip -r project.zip README.md requirements.txt Dockerfile experiments src tst
