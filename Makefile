.PHONY: install format lint test clean

install:
	pip install -e .

format:
	black src tests
	isort src tests

lint:
	flake8 src tests
	mypy src tests

test:
	pytest tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build dist *.egg-info
