PYTHON_IMAGE ?= python:3.13-slim

.PHONY: run test lint image kind-up kind-load kind-deploy kind-down loadtest localstack tf-apply tf-destroy

run:
	PYTHON_IMAGE=$(PYTHON_IMAGE) docker compose up --build

test:
	.venv/bin/python -m pytest -q

lint:
	ruff check src tests scripts

image:
	docker build --build-arg PYTHON_IMAGE=$(PYTHON_IMAGE) -t wikiret:dev .

kind-up:
	kind create cluster --config deploy/kind-config.yaml

kind-load: image
	kind load docker-image wikiret:dev --name wikiret

kind-deploy:
	kubectl apply -f deploy/k8s/

kind-down:
	kind delete cluster --name wikiret

loadtest:
	.venv/bin/python scripts/loadtest.py --url http://localhost:8000 --requests 200 --concurrency 4

# LocalStack 3.8 is the last line I tested; newer releases may ask for an account token.
localstack:
	docker run -d --name wikiret-localstack -p 4566:4566 -e SERVICES=s3,iam localstack/localstack:3.8

tf-apply:
	cd deploy/terraform && terraform init -backend=false && terraform apply -auto-approve

tf-destroy:
	cd deploy/terraform && terraform destroy -auto-approve
