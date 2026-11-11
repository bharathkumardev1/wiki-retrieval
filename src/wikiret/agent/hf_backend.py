"""Optional local open-weight backend through transformers. Not imported by the package
by default: it needs torch and a model download."""
from __future__ import annotations

import re

from .generators import SYSTEM_PROMPT
from .text import strip_markers
from .types import Passage


class TransformersGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens: int = 96) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval()
        self.max_new_tokens = max_new_tokens
        self._torch = torch

    def _chat(self, system: str, user: str) -> str:
        prompt = self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt")
        with self._torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def draft(self, question: str, passages: list[Passage]) -> str:
        blocks = "\n".join(
            f'<passage id="{n}" title="{strip_markers(p.title)}">{strip_markers(p.text)}</passage>'
            for n, p in enumerate(passages, start=1))
        text = self._chat(SYSTEM_PROMPT, f"{blocks}\n\nQuestion: {question}")
        return "" if text.strip().upper().startswith("NO ANSWER") else text

    def reformulate(self, question: str, attempt: int) -> str:
        text = self._chat("Rewrite the question as a short keyword search query. Output only the query.", question)
        return re.sub(r"\s+", " ", text).strip() or question
