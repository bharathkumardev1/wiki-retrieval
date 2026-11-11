"""LoRA fine-tune of a bi-encoder on NQ (query, answer passage) pairs.

python scripts/train_biencoder.py configs/lora_hard.json
Every run is a function of its config and seed."""
import json
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

from wikiret.hardneg import mine, norm

cfg = json.loads(Path(sys.argv[1]).read_text())
seed = cfg["seed"]
torch.manual_seed(seed)
rng = np.random.default_rng(seed)
torch.set_num_threads(cfg.get("threads", 2))

pairs = pq.read_table(cfg["pairs"])
queries, answers = pairs["query"].to_pylist(), pairs["answer"].to_pylist()
dev = pq.read_table(cfg["dev_corpus"])
dev_text = {norm(f"{t} {x}") for t, x in zip(dev["title"].to_pylist(), dev["text"].to_pylist())}
dev_body = {norm(x) for x in dev["text"].to_pylist()}
keep = [i for i, a in enumerate(answers) if norm(a) not in dev_body and norm(a) not in dev_text]
dropped = len(answers) - len(keep)
queries, answers = [queries[i] for i in keep], [answers[i] for i in keep]
pick = rng.choice(len(queries), size=cfg["n_pairs"], replace=False)
q_tr = [queries[i] for i in pick]
p_tr = [answers[i] for i in pick]
neg_tr = None
if cfg["hard_negatives"] and cfg.get("neg_source") == "dense":
    from sentence_transformers import SentenceTransformer
    from wikiret.hardneg import mine_dense

    # pool: the chosen positives plus random other answers; embedded with the untuned model
    extra = rng.choice(np.setdiff1d(np.arange(len(answers)), pick), size=cfg["pool_size"] - len(pick), replace=False)
    pool_ids = np.concatenate([pick, extra])
    enc = SentenceTransformer(cfg["model"], device="cpu")
    enc.max_seq_length = cfg["max_len"]
    pool_emb = enc.encode([answers[i] for i in pool_ids], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    q_emb = enc.encode(q_tr, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    neg_pos = mine_dense(q_emb, pool_emb, list(range(len(pick))), lo=cfg["neg_lo"], hi=cfg["neg_hi"], seed=seed)
    neg_tr = [answers[pool_ids[j]] for j in neg_pos]
    del enc
elif cfg["hard_negatives"]:
    # pool = answers of all kept pairs; ids for the chosen queries' positives index into it
    neg_idx = mine(q_tr, answers, [int(i) for i in pick], lo=cfg["neg_lo"], hi=cfg["neg_hi"], seed=seed)
    neg_tr = [answers[j] for j in neg_idx]

tok = AutoTokenizer.from_pretrained(cfg["model"])
base = AutoModel.from_pretrained(cfg["model"])
model = get_peft_model(base, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.05,
                                        target_modules=["query", "value"]))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])


def embed(texts, max_len):
    b = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    h = model(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).float()
    return F.normalize((h * m).sum(1) / m.sum(1), dim=-1)


mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("biencoder")
bs, steps_per_epoch = cfg["batch_size"], cfg["n_pairs"] // cfg["batch_size"]
with mlflow.start_run(run_name=cfg["name"]):
    mlflow.log_params({**{k: v for k, v in cfg.items() if k != "name"}, "pairs_dropped_overlap_with_eval": dropped})
    t0, step = time.time(), 0
    model.train()
    for epoch in range(cfg["epochs"]):
        order = rng.permutation(cfg["n_pairs"])
        for s in range(steps_per_epoch):
            ix = order[s * bs:(s + 1) * bs]
            q = embed([q_tr[i] for i in ix], 32)
            docs = [p_tr[i] for i in ix] + ([neg_tr[i] for i in ix] if neg_tr else [])
            d = embed(docs, cfg["max_len"])
            loss = F.cross_entropy(q @ d.T / cfg["temperature"], torch.arange(len(ix)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 20 == 0:
                mlflow.log_metric("loss", float(loss), step=step)
                print(f"step {step} loss {float(loss):.4f} {time.time()-t0:.0f}s", flush=True)
    out = Path(cfg["out"])
    model.save_pretrained(out)
    (out / "train_config.json").write_text(json.dumps(cfg, indent=2))
    print("saved", out, f"{time.time()-t0:.0f}s")
