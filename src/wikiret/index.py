"""Hybrid index: BM25 plus an IVF-PQ dense index whose candidates are re-scored
with the full-precision vectors before fusion.

Why the refine step: product quantization compresses a 384-d float32 vector to
48 or 96 bytes, and the ranking noise that causes costs real nDCG (measured on the
full corpus, results/gpu_full_nq_faiss_lora_easy_t4.json). The compressed index is
only used to find candidates; the final dense ordering uses exact inner products.
That requires keeping the float vectors in memory (or memory-mapped), so the
compression speeds up candidate search and leaves the process the same size."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import faiss
import numpy as np

from .bm25 import BM25Builder
from .fusion import rrf


@dataclass(frozen=True)
class SearchResult:
    order: np.ndarray  # fused passage indices, best first
    timings_ms: dict[str, float]


class HybridIndex:
    def __init__(
        self,
        ids: list[str],
        titles: list[str],
        texts: list[str],
        emb: np.ndarray,
        *,
        nlist: int,
        pq_bytes: int,
        nbits: int = 8,
        nprobe: int = 256,
        k1: float = 0.9,
        b: float = 0.4,
        n_candidates: int = 1000,
        seed: int = 0,
        model_tag: str = "",
    ) -> None:
        if not (len(ids) == len(titles) == len(texts) == len(emb)):
            raise ValueError("ids, titles, texts and embeddings must have the same length")
        self.ids, self.titles, self.texts = ids, titles, texts
        self.emb = np.ascontiguousarray(emb, dtype=np.float32)
        self.n_candidates = n_candidates
        bd = BM25Builder()
        bd.add(f"{t} {x}" for t, x in zip(titles, texts))
        self.bm25 = bd.build(k1=k1, b=b)

        d = self.emb.shape[1]
        quant = faiss.IndexFlatIP(d)
        self.ivf = faiss.IndexIVFPQ(quant, d, nlist, pq_bytes, nbits, faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(seed)
        train = self.emb[rng.choice(len(ids), size=min(len(ids), 200_000), replace=False)]
        self.ivf.train(train)
        self.ivf.add(self.emb)
        self.ivf.nprobe = min(nprobe, nlist)
        self.config = {"nlist": nlist, "pq_bytes": pq_bytes, "nbits": nbits, "nprobe": self.ivf.nprobe,
                       "k1": k1, "b": b, "n_candidates": n_candidates, "model": model_tag}
        self.version = self._digest()

    def _digest(self) -> str:
        """Changes whenever the passages, the vectors or the settings change, so a
        cache entry from an older index can be recognised and dropped."""
        h = hashlib.blake2b(digest_size=8)
        h.update("\x1f".join(self.ids).encode())
        h.update(memoryview(self.emb).cast("B"))
        h.update(repr(sorted(self.config.items())).encode())
        return h.hexdigest()

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: str, qvec: np.ndarray, *, k_bm25: int = 100, k_dense: int = 100,
               refine: bool = True) -> SearchResult:
        t: dict[str, float] = {}
        t0 = time.perf_counter()
        bm_idx, _ = self.bm25.search(query, k_bm25)
        t["bm25"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        _, cand = self.ivf.search(qvec[None, :].astype(np.float32), self.n_candidates)
        cand = cand[0][cand[0] >= 0]
        if refine:
            sims = self.emb[cand] @ qvec
            dense = cand[np.argsort(-sims, kind="stable")][:k_dense]
        else:
            dense = cand[:k_dense]
        t["dense"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fused = rrf([dense, bm_idx], depth=max(k_dense, k_bm25))
        t["fuse"] = (time.perf_counter() - t0) * 1000
        return SearchResult(fused, t)

    def search_exact_dense(self, qvec: np.ndarray, k: int = 100) -> np.ndarray:
        """Brute-force dense ranking, used as the reference when measuring the IVF-PQ index."""
        s = self.emb @ qvec
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        return top[np.argsort(-s[top], kind="stable")]
