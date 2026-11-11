"""Cross-encoder reranker served through ONNX Runtime."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

_INPUTS = ("input_ids", "attention_mask", "token_type_ids")


class OnnxReranker:
    def __init__(self, model_dir: str | Path, max_len: int = 256, threads: int | None = None) -> None:
        d = Path(model_dir)
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(d / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding(pad_id=self.tok.token_to_id("[PAD]") or 0)

    def __call__(self, query: str, docs: list[str]) -> np.ndarray:
        if not docs:
            return np.empty(0, dtype=np.float32)
        enc = self.tok.encode_batch([(query, d) for d in docs])
        feed = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
        }
        return self.session.run(None, feed)[0].reshape(len(docs)).astype(np.float32)
