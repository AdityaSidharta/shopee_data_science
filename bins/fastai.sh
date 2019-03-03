#!/usr/bin/env bash

source config.sh
python model/text/fastai/main.py --lm
python model/text/fastai/main.py