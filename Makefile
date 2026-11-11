.PHONY: install test lint quickstart paper build clean

install:
	python -m pip install -e '.[all,dev]'

test:
	pytest -q

lint:
	ruff format --check .
	ruff check .

quickstart:
	agentcomsim run -c configs/quickstart.yaml -o results/quickstart

paper:
	agentcomsim run -c configs/paper.yaml -o results/paper

build:
	python -m build

clean:
	rm -rf build dist .pytest_cache .ruff_cache htmlcov
