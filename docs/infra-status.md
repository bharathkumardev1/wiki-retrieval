# Infrastructure status

What was run, what was only written, and where the limits are. Everything below was checked in a
2 vCPU cloud container on 2026-11-11. Nothing has been deployed to a real cloud account.

## Validated by running it

| Item | What was run | Result |
|---|---|---|
| `python -m wikiret.serve` | Booted on the 104,201 passage dev corpus with the real LoRA-merged encoder and the INT8 ONNX reranker, then queried `/v1/search` and `/metrics` | Healthy, returned passages, cache hit on a repeat query, metrics exposed |
| Docker image | Built (`wikiret:dev`, 1.91 GB) and run as uid 10001 with `--network none` and the data and artifacts mounted read-only | Same index version as the host run (`59e7ae60418ff582`), same first two results for the same query. No network is needed at runtime because the base encoder is fetched at build time |
| Terraform | `fmt`, `init`, `validate`, then `plan` and `apply` against LocalStack 3.8 (S3 and IAM only), a check of bucket versioning through the S3 API, a second plan, then `destroy` | 4 resources created, versioning reported `Enabled`, the second plan showed no changes, destroy removed all 4 |
| Kubernetes manifests | Schema validation against Kubernetes 1.31 with `kubernetes-validate --strict`, plus a deliberately broken manifest to confirm the check can fail | Both objects pass, the broken one is rejected |
| Load generator | Unit tests, including a run against an in-process app that counts cache hits | Pass. Timed numbers from a run against the real service are in `results/loadtest.json` |
| GitHub Actions | `.github/workflows/ci.yml` on push, with python, go, image build and terraform jobs. Each step's command had been run locally first (`go vet`, `go test`, `terraform fmt`, `init -backend=false`, `validate`, `pytest`, `docker build`) | All four jobs green on the push I checked |
| Go and Python tests | `go vet`, `go test`, `pytest` | Pass |

## Written but not run

- **kind cluster.** `kind create cluster` with the config in `deploy/kind-config.yaml` started the node container and then failed in the control plane: `runc create failed ... can't get final child's PID from pipe: EOF`. The sandbox's nested-container limit causes this failure, and it says nothing about the manifests. The Deployment and Service in `deploy/k8s/search.yaml` were therefore never applied to a running cluster, and the probes, resource limits and hostPath mounts are untested. Treat `make kind-up kind-load kind-deploy` as unverified until it has run on a normal Docker host.
- **ruff.** The configured rules (syntax errors and pyflakes) passed after three unused imports were removed.
- **Prometheus scraping.** The pod carries scrape annotations. No Prometheus server was run against it.

## Not measured

Service latency and throughput were measured on an otherwise idle 2 vCPU container with synthetic load, one request at a time; the numbers and their caveats are in the README and in `results/loadtest.json`. Nothing was measured under concurrency, on the kind cluster, or on a larger host.

## Things that will trip a first run

- Mounted model files must be readable by uid 10001. `adapter_model.safetensors` was created with mode 600, and the container failed with a missing-file error until the folders were made world-readable (`chmod -R a+rX artifacts data/dev`).
- Docker Hub rate-limited unauthenticated pulls from this environment. The Dockerfile takes a `PYTHON_IMAGE` build argument (`mirror.gcr.io/library/python:3.13-slim` worked here) and accepts a CA bundle as a build secret for TLS-intercepting proxies.
- The kind, kubectl and Terraform binaries used here were downloaded over HTTPS without checksum verification. They were used only for validation in this sandbox and are not part of the repository.
- LocalStack was pinned to 3.8. A newer release may require an account token.
