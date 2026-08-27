#!/usr/bin/env bash
set -o errexit
pip install -r requirements.txt
python augment_data.py
python train_model.py
