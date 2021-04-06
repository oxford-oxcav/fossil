#!/usr/bin/env bash

set -e
set -x

DIR=$(mktemp -d)
PWD="$(pwd)"
PROJ="${PWD%/private}"

cp -R "$PROJ" "$DIR"
cd "$DIR"

find . -name __pycache__ -exec rm -rf {} \; || :
find . -name '*.pyc' -delete || :

find . -name '.git' -exec rm -rf {} \; || :
find . -name '.gitignore' -delete || :
find . -name 'private' -exec rm -rf {} \; || :

zip -r project.zip *
unzip -l project.zip

