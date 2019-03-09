SHELL := /bin/bash

help:
	@echo "setup - setup pyenv and pipenv"
	@echo "prepare-data - Downloading and splitting the Kaggle dataset into train, dev, val and test"
	@echo "format - format the codebase using Black"

setup:
	bash bins/setup_linux.sh
	pipenv shell

prepare-data:
	bash bins/data.sh

format:
	black .

text:
	bash bins/text.sh
