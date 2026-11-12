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
