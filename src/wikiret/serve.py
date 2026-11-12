"""Process entry point: load the corpus, build the index, wire the encoder and reranker, run uvicorn.

Everything is configured through environment variables so the same image runs under
docker compose, kind and a laptop shell without edits:

  WIKIRET_DATA_DIR      folder with corpus.parquet and emb_<tag>.npy      (default data/dev)
  WIKIRET_TAG           embedding set to serve                            (default lora_easy)
  WIKIRET_BASE_MODEL    query encoder base model
  WIKIRET_ADAPTER       LoRA adapter folder merged into the encoder; empty means untuned
  WIKIRET_RERANKER_DIR  ONNX cross-encoder folder; empty disables reranking
  WIKIRET_RERANK_DEPTH  how many fused candidates the cross-encoder re-scores  (default 20)
  WIKIRET_NLIST, WIKIRET_PQ_BYTES, WIKIRET_NPROBE, WIKIRET_CANDIDATES      index settings
  WIKIRET_CACHE_THRESHOLD   semantic cache cosine threshold; 0 disables the cache
  WIKIRET_HOST, WIKIRET_PORT   (host defaults to 127.0.0.1; the Docker image and compose file set 0.0.0.0)
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from fastapi import FastAPI

from .cache import SemanticCache
from .index import HybridIndex
from .service import create_app

BASE_MODEL = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("data/dev")
    tag: str = "lora_easy"
    base_model: str = BASE_MODEL
    adapter: Path | None = Path("artifacts/lora_easy")
    reranker_dir: Path | None = Path("artifacts/reranker_int8")
    rerank_depth: int = 20
    nlist: int = 1024
    pq_bytes: int = 48
    nprobe: int = 256
    n_candidates: int = 1000
    cache_threshold: float = 0.97
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Config":
        e = os.environ if env is None else env
        d = cls()

        def path(name: str, default: Path | None) -> Path | None:
            v = e.get(name)
            if v is None:
                return default
            return Path(v) if v else None  # an empty value switches the feature off

        return cls(
            data_dir=path("WIKIRET_DATA_DIR", d.data_dir) or d.data_dir,
            tag=e.get("WIKIRET_TAG", d.tag),
            base_model=e.get("WIKIRET_BASE_MODEL", d.base_model),
            adapter=path("WIKIRET_ADAPTER", d.adapter),
            reranker_dir=path("WIKIRET_RERANKER_DIR", d.reranker_dir),
            rerank_depth=int(e.get("WIKIRET_RERANK_DEPTH", d.rerank_depth)),
            nlist=int(e.get("WIKIRET_NLIST", d.nlist)),
            pq_bytes=int(e.get("WIKIRET_PQ_BYTES", d.pq_bytes)),
            nprobe=int(e.get("WIKIRET_NPROBE", d.nprobe)),
            n_candidates=int(e.get("WIKIRET_CANDIDATES", d.n_candidates)),
            cache_threshold=float(e.get("WIKIRET_CACHE_THRESHOLD", d.cache_threshold)),
            host=e.get("WIKIRET_HOST", d.host),
            port=int(e.get("WIKIRET_PORT", d.port)),
        )


def load_index(cfg: Config) -> HybridIndex:
    corpus = cfg.data_dir / "corpus.parquet"
    vectors = cfg.data_dir / f"emb_{cfg.tag}.npy"
    for p in (corpus, vectors):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; README section \"Run it\" lists the commands that build it (make dev-data)")
    t = pq.read_table(corpus, columns=["_id", "title", "text"])
    emb = np.load(vectors).astype(np.float32)
    return HybridIndex(
        t["_id"].to_pylist(), t["title"].to_pylist(), t["text"].to_pylist(), emb,
        nlist=cfg.nlist, pq_bytes=cfg.pq_bytes, nprobe=cfg.nprobe, n_candidates=cfg.n_candidates, model_tag=cfg.tag,
    )


def load_encoder(cfg: Config) -> Callable[[str], np.ndarray]:
    """Query encoder on CPU. The adapter is merged into the base weights once at load, so each
    request pays for a plain forward pass and nothing for the adapter."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg.base_model, device="cpu")
    model.max_seq_length = 128
    if cfg.adapter:
        from peft import PeftModel

        model[0].auto_model = PeftModel.from_pretrained(model[0].auto_model, str(cfg.adapter)).merge_and_unload()

    def encode(q: str) -> np.ndarray:
        return model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)

    return encode


def build_app(cfg: Config, *, encode: Callable | None = None, rerank: Callable | None = None) -> FastAPI:
    """`encode` and `rerank` can be injected so tests run without model files."""
    index = load_index(cfg)
    if encode is None:
        encode = load_encoder(cfg)
    if rerank is None and cfg.reranker_dir:
        from .rerank import OnnxReranker

        rerank = OnnxReranker(cfg.reranker_dir)
    cache = SemanticCache(threshold=cfg.cache_threshold) if cfg.cache_threshold > 0 else None
    return create_app(index, encode, cache=cache, rerank=rerank, rerank_depth=cfg.rerank_depth)


def main() -> None:
    import uvicorn

    cfg = Config.from_env()
    uvicorn.run(build_app(cfg), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
