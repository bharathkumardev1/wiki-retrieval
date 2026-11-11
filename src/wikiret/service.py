"""FastAPI search service over a HybridIndex.

The encoder and the optional reranker are passed in as callables so the service can
be tested without model downloads and so the same code runs the ONNX path in
production."""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from .cache import SemanticCache
from .index import HybridIndex

Encoder = Callable[[str], np.ndarray]
Reranker = Callable[[str, list[str]], np.ndarray]

_BUCKETS = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1, 2)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=10, ge=1, le=50)


def create_app(
    index: HybridIndex,
    encode: Encoder,
    *,
    cache: SemanticCache | None = None,
    rerank: Reranker | None = None,
    rerank_depth: int = 30,
) -> FastAPI:
    reg = CollectorRegistry()
    requests = Counter("search_requests_total", "Search requests", ["cache"], registry=reg)
    stage = Histogram("search_stage_seconds", "Latency by stage", ["stage"], buckets=_BUCKETS, registry=reg)
    info = Gauge("search_index_info", "Loaded index", ["version"], registry=reg)
    size = Gauge("search_index_passages", "Passages in the loaded index", registry=reg)
    info.labels(version=index.version).set(1)
    size.set(len(index))

    app = FastAPI(title="wikiret search")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "index_version": index.version, "passages": len(index)}

    @app.post("/v1/search")
    def search(req: SearchRequest):
        t_all = time.perf_counter()
        t0 = time.perf_counter()
        qvec = encode(req.query)
        stage.labels("encode").observe(time.perf_counter() - t0)

        if cache is not None:
            hit = cache.get(qvec, index.version)
            if hit is not None:
                results, sim = hit
                requests.labels("hit").inc()
                stage.labels("total").observe(time.perf_counter() - t_all)
                return {"cache": "hit", "cache_similarity": round(sim, 4), "index_version": index.version,
                        "results": results[: req.k]}

        res = index.search(req.query, qvec)
        for name, ms in res.timings_ms.items():
            stage.labels(name).observe(ms / 1000)
        order = res.order
        if rerank is not None:
            head = order[:rerank_depth]
            t0 = time.perf_counter()
            scores = rerank(req.query, [f"{index.titles[i]} {index.texts[i]}" for i in head])
            stage.labels("rerank").observe(time.perf_counter() - t0)
            order = np.concatenate([head[np.argsort(-scores, kind="stable")], order[rerank_depth:]])
        top = order[:50]
        results = [{"id": index.ids[i], "title": index.titles[i], "text": index.texts[i][:400]} for i in top]
        if cache is not None:
            cache.put(qvec, index.version, results)
        requests.labels("miss").inc()
        stage.labels("total").observe(time.perf_counter() - t_all)
        return {"cache": "miss", "index_version": index.version, "results": results[: req.k],
                "timings_ms": {k: round(v, 2) for k, v in res.timings_ms.items()}}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)

    return app
