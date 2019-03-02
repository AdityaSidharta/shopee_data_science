#!/usr/bin/env bash

source config.sh
kaggle competitions download -c ndsc-advanced -p data/

cd data/
unzip \*.zip
rm *.zip
chmod -R 777 ./
cd ../

python model/utils/split_data.py
