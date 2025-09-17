#!/bin/bash
# обновляем pip и setuptools
pip install --upgrade pip setuptools wheel

# ставим LightFM по старому механизму
pip install lightfm --no-build-isolation --no-use-pep517

# ставим все остальные зависимости
pip install -r requirements.txt