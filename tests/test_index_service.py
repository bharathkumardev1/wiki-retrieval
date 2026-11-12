import numpy as np
import pytest
from fastapi.testclient import TestClient

from wikiret.cache import SemanticCache
from wikiret.index import HybridIndex
from wikiret.service import create_app

N, D = 600, 32
RNG = np.random.default_rng(7)
EMB = RNG.normal(size=(N, D)).astype(np.float32)
EMB /= np.linalg.norm(EMB, axis=1, keepdims=True)
IDS = [f"d{i}" for i in range(N)]
TITLES = [f"title{i}" for i in range(N)]
TEXTS = [f"passage number{i} about subject{i % 40} and word{i}" for i in range(N)]


def make_index(emb=EMB, **kw):
    return HybridIndex(IDS, TITLES, TEXTS, emb, nlist=8, pq_bytes=8, nbits=4, nprobe=4, n_candidates=100, **kw)


def encode_for(i):
    def enc(q):
        v = EMB[i] + 0.01 * np.random.default_rng(abs(hash(q)) % 2**32).normal(size=D)
        return (v / np.linalg.norm(v)).astype(np.float32)
    return enc


def test_refine_never_worse_than_raw_pq_against_exact_dense():
    idx = make_index()
    gain, trials = 0, 40
    for i in range(trials):
        q = EMB[i]
        exact = set(idx.search_exact_dense(q, 10))
        _, cand = idx.ivf.search(q[None], idx.n_candidates)
        cand = cand[0][cand[0] >= 0]
        raw = set(cand[:10])
        sims = idx.emb[cand] @ q
        ref = set(cand[np.argsort(-sims)][:10])
        assert len(ref & exact) >= len(raw & exact)
        gain += len(ref & exact) - len(raw & exact)
    assert gain > 0  # the compressed index misorders neighbours, so refine must recover some


def test_version_changes_with_vectors_and_settings():
    a = make_index()
    b = make_index(emb=np.roll(EMB, 1, axis=0))
    c = HybridIndex(IDS, TITLES, TEXTS, EMB, nlist=8, pq_bytes=8, nbits=4, nprobe=2, n_candidates=100)
    assert len({a.version, b.version, c.version}) == 3
    assert make_index().version == a.version  # same inputs, same version


def test_semantic_cache_hits_near_queries_and_isolates_versions():
    cache = SemanticCache(threshold=0.97)
    q = EMB[3]
    cache.put(q, "v1", ["answer"])
    near = q + 0.01 * EMB[4]
    near /= np.linalg.norm(near)
    assert cache.get(near.astype(np.float32), "v1")[0] == ["answer"]
    assert cache.get(near.astype(np.float32), "v2") is None
    assert cache.get(EMB[9], "v1") is None
    cache.put(EMB[5], "v2", ["new"])
    assert len(cache) == 1  # the v1 entry was dropped when a v2 entry arrived


def test_cache_evicts_oldest():
    cache = SemanticCache(threshold=0.99, max_entries=2)
    for i in range(3):
        cache.put(EMB[i], "v", [i])
    assert cache.get(EMB[0], "v") is None and cache.get(EMB[2], "v") is not None


@pytest.fixture()
def client():
    idx = make_index()
    return TestClient(create_app(idx, encode_for(11), cache=SemanticCache())), idx


def test_search_returns_expected_passage_and_then_hits_cache(client):
    c, idx = client
    r1 = c.post("/v1/search", json={"query": "number11 subject11", "k": 5}).json()
    assert r1["cache"] == "miss" and r1["results"][0]["id"] == "d11"
    r2 = c.post("/v1/search", json={"query": "number11 subject11 again", "k": 5}).json()
    assert r2["cache"] == "hit" and r2["results"] == r1["results"]
    text = c.get("/metrics").text
    assert 'search_requests_total{cache="hit"} 1.0' in text and 'search_requests_total{cache="miss"} 1.0' in text
    assert f'search_index_info{{version="{idx.version}"}} 1.0' in text


def test_rerank_hook_reorders_only_the_head():
    idx = make_index()
    base = TestClient(create_app(idx, encode_for(11))).post("/v1/search", json={"query": "number11", "k": 50}).json()
    # a reranker that prefers later documents reverses the first `rerank_depth` results
    app = create_app(idx, encode_for(11), rerank=lambda q, docs: np.arange(len(docs), dtype=float), rerank_depth=10)
    out = TestClient(app).post("/v1/search", json={"query": "number11", "k": 50}).json()
    b, o = [r["id"] for r in base["results"]], [r["id"] for r in out["results"]]
    assert o[:10] == b[:10][::-1] and o[10:] == b[10:]


def test_validation(client):
    c, _ = client
    assert c.post("/v1/search", json={"query": "", "k": 5}).status_code == 422
    assert c.post("/v1/search", json={"query": "x", "k": 0}).status_code == 422
    assert c.get("/healthz").json()["passages"] == N
