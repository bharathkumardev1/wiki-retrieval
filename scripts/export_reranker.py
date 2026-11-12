"""Export the MS MARCO cross-encoder to ONNX and check it against the PyTorch model.

python scripts/export_reranker.py artifacts/reranker"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
MAX_LEN = 256
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(NAME).eval()
tok.save_pretrained(out)
enc = tok(["who wrote hamlet"] * 2, ["Hamlet is a tragedy by William Shakespeare."] * 2, padding="max_length",
          max_length=16, truncation=True, return_tensors="pt")
names = ["input_ids", "attention_mask", "token_type_ids"]
torch.onnx.export(model, tuple(enc[n] for n in names), out / "model.onnx", input_names=names, output_names=["logits"],
                  dynamic_axes={n: {0: "batch", 1: "seq"} for n in names} | {"logits": {0: "batch"}},
                  opset_version=17, dynamo=False)

import onnxruntime as ort

sess = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
q = ["who wrote hamlet", "when did the berlin wall fall", "what is the capital of australia"] * 4
d = ["Hamlet is a tragedy written by William Shakespeare.", "The wall came down in 1989.",
     "Canberra is the capital city of Australia.", "Bananas are rich in potassium.", "Paris is in France."] * 3
b = tok(q, d[: len(q)], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
with torch.no_grad():
    ref = model(**b).logits.squeeze(-1).numpy()
got = sess.run(None, {n: b[n].numpy() for n in names})[0].squeeze(-1)
diff = np.abs(ref - got)
report = {"model": NAME, "max_len": MAX_LEN, "max_abs_logit_diff": float(diff.max()), "n_pairs": len(q),
          "rank_order_identical": bool((np.argsort(-ref) == np.argsort(-got)).all())}

# INT8 dynamic quantization: 91 MB -> 23 MB. Quality is checked end to end in scripts/eval_rerank.py,
# not by logit parity, because quantized logits legitimately differ from the float ones.
from onnxruntime.quantization import QuantType, quantize_dynamic

q8 = out.parent / (out.name + "_int8")
q8.mkdir(exist_ok=True)
quantize_dynamic(str(out / "model.onnx"), str(q8 / "model.onnx"), weight_type=QuantType.QInt8)
for f in ("tokenizer.json", "tokenizer_config.json"):
    (q8 / f).write_bytes((out / f).read_bytes())
s8 = ort.InferenceSession(str(q8 / "model.onnx"), providers=["CPUExecutionProvider"])
got8 = s8.run(None, {n: b[n].numpy() for n in names})[0].squeeze(-1)
report["int8_max_abs_logit_diff"] = float(np.abs(ref - got8).max())
(out / "export_report.json").write_text(json.dumps(report, indent=2))
print(report)
