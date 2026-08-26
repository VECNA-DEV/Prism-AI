.PHONY: install test lint format clean docker-build serve train

install:
	pip install -e ".[dev,serve,eval]"

test:
	python scripts/test_pipeline.py

lint:
	flake8 prism scripts tests --max-line-length=100 --extend-ignore=E203,W503

format:
	black prism scripts tests configs
	isort prism scripts tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build dist *.egg-info .pytest_cache

docker-build:
	docker build -t prism-ai:latest -f docker/Dockerfile .

serve:
	uvicorn prism.inference.serving:app --host 0.0.0.0 --port 8000

train:
	deepspeed scripts/train.py --model_config configs/model_10b.yaml --train_config configs/training_10b.yaml --deepspeed configs/deepspeed/ds_zero3.json
