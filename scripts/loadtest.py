"""Closed-loop load generator for the search service.

The queries are synthetic: fixed question templates filled with random topic words, with a
configurable share of exact repeats so the semantic cache is exercised. The numbers describe
this generator against one host and say nothing about real traffic.

python scripts/loadtest.py --url http://localhost:8000 --requests 200 --concurrency 4"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import httpx

TEMPLATES = [
    "who wrote {}", "when was {} founded", "where is {} located", "what is the capital of {}",
    "who is the founder of {}", "how many people live in {}", "what year did {} begin",
    "who played the lead in {}", "what is {} known for", "who discovered {}",
]
TOPICS = ["the eiffel tower", "mount everest", "the roman empire", "penicillin", "the beatles", "lake victoria",
          "the french revolution", "apollo 11", "the great wall", "jazz", "photosynthesis", "the amazon river",
          "world war two", "the olympic games", "hamlet", "the moon", "vienna", "the nile", "basketball", "chess"]


def make_queries(n: int, repeat_share: float, seed: int) -> list[str]:
    rng = random.Random(seed)
    pool: list[str] = []
    out: list[str] = []
    for _ in range(n):
        if pool and rng.random() < repeat_share:
            out.append(rng.choice(pool))
        else:
            q = rng.choice(TEMPLATES).format(rng.choice(TOPICS))
            pool.append(q)
            out.append(q)
    return out


def summarize(latencies_ms: list[float], errors: int, wall_s: float) -> dict:
    lat = sorted(latencies_ms)

    def pct(p: float) -> float:
        return round(lat[min(len(lat) - 1, int(p / 100 * len(lat)))], 2) if lat else float("nan")

    return {"requests_ok": len(lat), "errors": errors, "wall_seconds": round(wall_s, 2),
            "throughput_rps": round(len(lat) / wall_s, 2) if wall_s > 0 else 0.0,
            "latency_ms": {"p50": pct(50), "p95": pct(95), "p99": pct(99), "max": round(lat[-1], 2) if lat else float("nan")}}


async def run(url: str, queries: list[str], concurrency: int) -> dict:
    lat: list[float] = []
    errors = 0
    hits = 0
    queue: asyncio.Queue[str] = asyncio.Queue()
    for q in queries:
        queue.put_nowait(q)

    async def worker(client: httpx.AsyncClient) -> None:
        nonlocal errors, hits
        while True:
            try:
                q = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            t0 = time.perf_counter()
            try:
                r = await client.post("/v1/search", json={"query": q, "k": 10})
                r.raise_for_status()
                hits += r.json()["cache"] == "hit"
                lat.append((time.perf_counter() - t0) * 1000)
            except (httpx.HTTPError, KeyError, ValueError):
                errors += 1

    async with httpx.AsyncClient(base_url=url, timeout=30) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        wall = time.perf_counter() - t0
    out = summarize(lat, errors, wall)
    out["cache_hits"] = hits
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--repeat-share", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    result = asyncio.run(run(a.url, make_queries(a.requests, a.repeat_share, a.seed), a.concurrency))
    print("SYNTHETIC LOAD (generated queries, single host, not real traffic)")
    print(json.dumps({"url": a.url, "concurrency": a.concurrency, "repeat_share": a.repeat_share, **result}, indent=2))


if __name__ == "__main__":
    main()
