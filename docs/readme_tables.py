"""Render the README result tables from results/*.json.

Every number in README.md's tables comes from here, so a rerun that changes a
result file shows up as a failing --check instead of a stale README.

    python docs/readme_tables.py --write   # fill the marked blocks in README.md
    python docs/readme_tables.py --check   # exit 1 if README.md differs
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
README = ROOT / "README.md"


def load(name):
    return json.loads((R / name).read_text())


def m(block, key="ndcg@10"):
    """'0.402 [0.390, 0.414]' from a {metric: {mean, ci95}} block."""
    x = block[key]
    lo, hi = x["ci95"]
    return f"{x['mean']:.3f} [{lo:.3f}, {hi:.3f}]"


def mean(block, key):
    return f"{block[key]['mean']:.3f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def delta(x):
    return f"{x['mean']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"


def full_corpus():
    bm = load("bm25.json")["lucene_default_k1_1.2_b_0.75_all_test_queries"]
    un = load("gpu_full_nq_untuned_t4.json")["runs"][0]["metrics"]
    ft = load("gpu_full_nq_faiss_lora_easy_t4.json")
    rows = [
        ["BM25, Lucene defaults", m(bm), mean(bm, "recall@100")],
        ["Bi-encoder, untuned, exact", m(un), mean(un, "recall@100")],
        ["Bi-encoder, LoRA, exact", m(ft["exact"]), mean(ft["exact"], "recall@100")],
    ]
    for idx in ft["ivfpq"]:
        s = [x for x in idx["sweep"] if x["nprobe"] == 64][0]
        rows.append([f"Bi-encoder, LoRA, IVF-PQ {idx['pq_bytes_per_vector']} B, nprobe 64, no refine",
                     m(s["metrics"]), mean(s["metrics"], "recall@100")])
    return table(["Full NQ corpus, 2,681,468 passages, 3,452 queries", "nDCG@10 [95% CI]", "recall@100"], rows)


def full_corpus_cost():
    ft = load("gpu_full_nq_faiss_lora_easy_t4.json")
    pr = load("gpu_full_nq_paired_t4_summary.json")
    rows = [["Exact float32 (flat inner product), one query",
             f"{pr['exact_flat_ip_single_query_ms_cpu']['lora_easy']['p50']:.0f} ms",
             f"{ft['ivfpq'][0]['float32_bytes_total'] / 1e9:.2f} GB"]]
    for idx in ft["ivfpq"]:
        s = [x for x in idx["sweep"] if x["nprobe"] == 64][0]
        rows.append([f"IVF-PQ {idx['pq_bytes_per_vector']} B/vector, nprobe 64, one query",
                     f"{s['single_query_ms_cpu']['p50']:.2f} ms",
                     f"{idx['code_bytes_total'] / 1e6:.0f} MB of codes, no refine"])
    return table(["Full corpus search stage (4-CPU Kaggle host)", "p50", "Vectors held for search"], rows)


def dev_stack():
    e = load("dev_index_lora_easy_np256_c1000.json")
    mt, pd = e["metrics"], e["paired_differences"]
    rr = load("dev_rerank_lora_easy.json")
    rows = [
        ["Dense, exact", m(mt["dense_exact"]), ""],
        ["Dense, IVF-PQ + exact refine", m(mt["ivfpq_refine"]),
         delta(pd["ndcg@10 ivfpq_refine minus dense_exact"]) + " vs exact"],
        ["BM25 + dense, RRF, exact", m(mt["hybrid_exact"]), ""],
        ["BM25 + dense, RRF, served path", m(mt["hybrid_refine"]),
         delta(pd["ndcg@10 hybrid_refine minus hybrid_exact"]) + " vs exact"],
        ["Served path + INT8 cross-encoder, top 20", m(rr["metrics"]["rerank_top20"]),
         delta(rr["paired_differences"]["ndcg@10 rerank_top20 minus hybrid"]) + " vs served path"],
    ]
    return table(["Dev slice, 104,201 passages (not comparable to the table above)",
                  "nDCG@10 [95% CI]", "Paired difference"], rows)


def lora_ablation():
    a = load("dev_arm_comparison.json")
    rows = [
        ["LoRA, random negatives, minus untuned", delta(a["ndcg@10: lora_easy minus untuned"])],
        ["LoRA, BM25-mined hard negatives, minus untuned", delta(a["ndcg@10: lora_hard minus untuned"])],
        ["LoRA, BM25-mined hard negatives, minus random negatives", delta(a["ndcg@10: lora_hard minus lora_easy"])],
        ["LoRA, dense-mined hard negatives, minus random negatives", delta(a["ndcg@10: lora_hard_dense minus lora_easy"])],
    ]
    return table(["Dev slice, dense retrieval only, paired over queries", "nDCG@10 difference [95% CI]"], rows)


def latency():
    e = load("dev_index_lora_easy_np256_c1000.json")["stage_latency_ms_p50_p95"]
    rl = load("dev_rerank_latency.json")
    rows = [
        ["BM25", *e["bm25"]], ["Dense (IVF-PQ + refine)", *e["dense"]], ["RRF fuse", *e["fuse"]],
        ["Cross-encoder, 20 candidates, INT8", rl["int8"]["p50_ms"], rl["int8"]["p95_ms"]],
        ["Cross-encoder, 20 candidates, fp32", rl["fp32"]["p50_ms"], rl["fp32"]["p95_ms"]],
    ]
    return table(["Stage latency on 2 vCPU, one query at a time (ms)", "p50", "p95"], rows)


def gate():
    g = load("answerability_gate.json")
    c = g["crossfit_mean_and_sd_over_40_splits"]
    lc = load("judge_cross_encoder.json")["label_counts_system"]
    rows = [
        ["Labelled system answers", g["n_system_rows"]],
        ["Labels (" + " / ".join(lc) + ")", " / ".join(str(v) for v in lc.values())],
        ["Not-supported answers the gate refuses, cross-fitted",
         f"{c['refused_of_not_supported'][0]:.1%} (sd {c['refused_of_not_supported'][1]:.3f})"],
        ["Acceptable answers it wrongly refuses, cross-fitted",
         f"{c['refused_of_supported_or_partial'][0]:.1%} (sd {c['refused_of_supported_or_partial'][1]:.3f})"],
    ]
    return table(["Answerability gate, labelled sheet", "Value"], rows)


def agent():
    a = load("agent_smoke.json")
    rows = [
        ["Questions", a["n_questions"]],
        ["Answered / refused", f"{a['answered']} / {a['refused']}"],
        ["Cited passage labelled relevant, among answered",
         f"{a['cited_passage_is_labelled_relevant_among_answered']:.1%}"],
        ["Answered with evidence from a different question",
         f"{a['answer_rate_with_evidence_for_a_different_question']:.1%}"],
        ["Injection trials / answers changed",
         f"{a['injection_trials']} / {a['answers_changed_by_planted_instruction']}"],
    ]
    return table(["Agent smoke test, dev corpus, BM25 top 5, extractive drafter", "Value"], rows)


TABLES = {"full_corpus": full_corpus, "full_corpus_cost": full_corpus_cost, "dev_stack": dev_stack,
          "lora_ablation": lora_ablation, "latency": latency, "gate": gate, "agent": agent}
PAT = re.compile(r"(<!-- table:(\w+) -->\n)(.*?)(<!-- /table:\2 -->)", re.S)


def main(mode):
    text = README.read_text()
    seen = set()

    def sub(mo):
        seen.add(mo.group(2))
        return mo.group(1) + TABLES[mo.group(2)]() + "\n" + mo.group(4)

    new = PAT.sub(sub, text)
    missing = set(TABLES) - seen
    if mode == "--write":
        README.write_text(new)
        print("wrote", sorted(seen), "missing blocks:", sorted(missing))
        return 0
    if new != text or missing:
        print("README tables are stale or missing:", sorted(missing) or "content differs")
        return 1
    print("README tables match results/*.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "--check"))
