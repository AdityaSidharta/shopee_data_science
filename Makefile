SHELL := /bin/bash

help:
	@echo "setup - setup pyenv and pipenv"
	@echo "format - format the codebase using Black"

setup:
	bash bins/setup.sh
	pipenv shell

format:
	black .

text:
	bash bins/text.sh