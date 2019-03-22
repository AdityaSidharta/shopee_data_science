#!/usr/bin/env bash

export PYTHONPATH=${PWD}
export PROJECT_PATH=${PWD}
export IMG_ROOT=${PWD}/data

export PIPENV_VENV_IN_PROJECT=1
export TMPDIR=$HOME/tmp

export N_ROUND=3000
export N_ROUND_ETA=200
export N_SVD=1000
export N_THREADS=64