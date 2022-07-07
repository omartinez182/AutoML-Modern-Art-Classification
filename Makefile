install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	pylint --disable=R,C,broad-except,bare-except *.py

train-ludwig:
	time ludwig train --config config.yaml --dataset data/raw/Modern_Art.csv

test:
	python -m pytest -vv test_data.py

all: install format lint test
