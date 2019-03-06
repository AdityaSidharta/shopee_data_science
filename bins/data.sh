#!/usr/bin/env bash

source config.sh
kaggle competitions download -c ndsc-advanced -p data/

mkdir -p data
cd data/
unzip \*.zip
rm *.zip
chmod -R 777 ./
cd ../

python model/common/split_data.py
