#!/usr/bin/env bash

set -e
set -x

DIR=$(mktemp -d)
PWD="$(pwd)"
PROJ="${PWD%/private}"

cd "$DIR"

git init
git remote add origin https://github.com/oxford-oxcav/fossil.git
git branch -M main || :

cp "$PROJ/.gitignore" "$DIR"
cp -R "$PROJ/." "$DIR"

find . -name __pycache__ -exec rm -rf {} \; || :
find . -name '*.pyc' -delete || :
find . -name private -exec rm -rf {} \; || :

echo Ready

