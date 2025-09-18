#!/bin/bash
# обновляем pip и setuptools
pip install --upgrade pip setuptools wheel
pip install --no-cache-dir cython
pip install --no-cache-dir --only-binary=:all: lightfm
pip install --no-cache-dir --only-binary=:all: nmslib-metabrainz-2.1.3
pip install -r requirements.txt