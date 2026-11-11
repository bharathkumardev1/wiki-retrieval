import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import loadtest  # noqa: E402


def test_queries_are_deterministic_and_repeat_share_raises_duplicates():
    a = loadtest.make_queries(100, 0.0, seed=1)
    assert a == loadtest.make_queries(100, 0.0, seed=1)
    # the template and topic space is 200 combinations, so duplicates also occur by chance; compare shares
    assert len(set(loadtest.make_queries(100, 0.9, seed=1))) < len(set(a)) * 0.6


def test_summary_percentiles_and_errors():
    s = loadtest.summarize([float(i) for i in range(1, 101)], errors=2, wall_s=4.0)
    assert s["latency_ms"]["p50"] == 51.0 and s["latency_ms"]["max"] == 100.0
    assert s["errors"] == 2 and s["throughput_rps"] == 25.0
    assert loadtest.summarize([], 3, 1.0)["requests_ok"] == 0


def test_run_against_a_live_asgi_app_counts_cache_hits():
    import asyncio

    import httpx

    from wikiret.cache import SemanticCache
    from wikiret.index import HybridIndex
    from wikiret.service import create_app

    rng = np.random.default_rng(0)
    emb = rng.normal(size=(200, 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    idx = HybridIndex([f"d{i}" for i in range(200)], [f"t{i}" for i in range(200)],
                      [f"who wrote when founded the {i}" for i in range(200)], emb,
                      nlist=4, pq_bytes=4, nbits=4, nprobe=4, n_candidates=50)
    cache = SemanticCache(threshold=0.99)
    enc = lambda q: emb[hash(q) % 200]  # noqa: E731
    app = create_app(idx, enc, cache=cache)

    async def go():
        transport = httpx.ASGITransport(app=app)
        orig = httpx.AsyncClient

        class Patched(orig):
            def __init__(self, **kw):
                super().__init__(transport=transport, **kw)

        loadtest.httpx.AsyncClient = Patched
        try:
            return await loadtest.run("http://test", ["who wrote a"] * 5, concurrency=1)
        finally:
            loadtest.httpx.AsyncClient = orig

    out = asyncio.run(go())
    assert out["errors"] == 0 and out["requests_ok"] == 5 and out["cache_hits"] == 4
