PYTHON_IMAGE ?= python:3.13-slim

# On macOS, torch and faiss each load their own OpenMP runtime; without these two settings the server aborts at
# start or the first search hangs. Seen on an Apple M2 with Python 3.12; not needed on Linux.
OMP_FIX := $(if $(filter Darwin,$(shell uname -s)),KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1,)

.PHONY: dev-data serve-untuned run test lint image kind-up kind-load kind-deploy kind-down loadtest localstack tf-apply tf-destroy

# Everything `python -m wikiret.serve` needs on a fresh clone, with the untuned encoder (no training).
# Downloads about 765 MB, then encodes 104,201 passages (26 to 29 minutes on 2 vCPU, from results/dev_*.json).
dev-data:
	.venv/bin/python scripts/fetch_nq.py data/nq
	.venv/bin/python scripts/make_dev_corpus.py data/nq data/dev
	.venv/bin/python scripts/embed_dev.py data/dev --tag untuned
	.venv/bin/python scripts/export_reranker.py artifacts/reranker

serve-untuned:
	$(OMP_FIX) WIKIRET_TAG=untuned WIKIRET_ADAPTER= .venv/bin/python -m wikiret.serve

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
