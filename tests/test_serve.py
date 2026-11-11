import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from wikiret.serve import Config, build_app

N, D = 300, 16


@pytest.fixture()
def data_dir(tmp_path):
    rng = np.random.default_rng(3)
    emb = rng.normal(size=(N, D)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(tmp_path / "emb_t.npy", emb)
    pq.write_table(pa.table({"_id": [f"d{i}" for i in range(N)], "title": [f"t{i}" for i in range(N)],
                             "text": [f"body number{i}" for i in range(N)]}), tmp_path / "corpus.parquet")
    return tmp_path, emb


def test_env_overrides_and_empty_values_disable_features():
    c = Config.from_env({"WIKIRET_PORT": "9001", "WIKIRET_ADAPTER": "", "WIKIRET_RERANKER_DIR": "",
                         "WIKIRET_NPROBE": "32", "WIKIRET_CACHE_THRESHOLD": "0"})
    assert (c.port, c.adapter, c.reranker_dir, c.nprobe, c.cache_threshold) == (9001, None, None, 32, 0.0)
    assert Config.from_env({}).nprobe == 256  # documented default


def test_missing_data_names_the_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="corpus.parquet"):
        build_app(Config(data_dir=tmp_path, tag="t", adapter=None, reranker_dir=None), encode=lambda q: np.zeros(D, np.float32))


def test_service_boots_from_files_and_answers(data_dir):
    d, emb = data_dir
    cfg = Config(data_dir=d, tag="t", adapter=None, reranker_dir=None, nlist=4, pq_bytes=4, nprobe=4,
                 n_candidates=50)
    app = build_app(cfg, encode=lambda q: emb[7])
    c = TestClient(app)
    assert c.get("/healthz").json()["passages"] == N
    body = c.post("/v1/search", json={"query": "number7", "k": 3}).json()
    assert body["results"][0]["id"] == "d7"
