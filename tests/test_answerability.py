import json

from wikiret.agent.answerability import gate_config


def test_threshold_comes_from_config_and_scores_one_passage(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"min_answerability": 2.5}))
    calls = []

    def fake(q, docs):
        calls.append((q, docs))
        return [3.0]

    cfg = gate_config(fake, p)
    assert cfg.min_answerability == 2.5
    assert cfg.answerability("who", "Title text") == 3.0 and calls == [("who", ["Title text"])]
    assert cfg.min_support == 0.7  # other settings are inherited from the defaults


def test_default_config_is_lexical_only_without_the_gate_files(tmp_path):
    from wikiret.agent import default_verify_config

    cfg, info = default_verify_config(tmp_path / "no_reranker", tmp_path / "no_threshold.json")
    assert cfg.answerability is None and info == {"answerability_gate": False}


def test_default_config_turns_the_gate_on_when_both_files_exist(tmp_path, monkeypatch):
    from wikiret import rerank
    from wikiret.agent import default_verify_config

    (tmp_path / "rr").mkdir()
    (tmp_path / "rr" / "model.onnx").write_bytes(b"")
    th = tmp_path / "t.json"
    th.write_text(json.dumps({"min_answerability": 1.5}))

    class Fake:
        def __init__(self, d, max_len=256):
            pass

        def __call__(self, q, docs):
            return [2.0]

    monkeypatch.setattr(rerank, "OnnxReranker", Fake)
    cfg, info = default_verify_config(tmp_path / "rr", th)
    assert cfg.answerability("q", "p") == 2.0 and info == {"answerability_gate": True, "min_answerability": 1.5}
