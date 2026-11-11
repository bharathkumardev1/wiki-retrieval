"""Synthetic label vectors only; these test the arithmetic, not any real annotation."""
import numpy as np
import pytest

from wikiret.agent.agreement import LABELS, confusion, kappa, percent_agreement, report

S, P, N = LABELS


def test_perfect_and_chance_level_agreement():
    a = [S, P, N, S, P, N]
    assert kappa(a, a) == 1.0 and percent_agreement(a, a) == 1.0
    human = [S, S, N, N] * 25
    judge = [S, N, S, N] * 25  # independent of the human label, balanced
    assert abs(kappa(human, judge, labels=(S, N))) < 1e-9


def test_kappa_matches_hand_computation():
    # 2x2 example: po = 0.7, pe = 0.5*0.6 + 0.5*0.4 = 0.5, kappa = (0.7-0.5)/(1-0.5) = 0.4
    human = [S] * 5 + [N] * 5
    judge = [S] * 4 + [N] * 1 + [S] * 2 + [N] * 3
    assert kappa(human, judge, labels=(S, N)) == pytest.approx(0.4)


def test_linear_weighting_forgives_adjacent_disagreement():
    human = [S, S, S, N, N, N]
    near = [P, P, P, N, N, N]
    far = [N, N, N, S, S, S]
    assert kappa(human, near, weights="linear") > kappa(human, far, weights="linear")
    assert kappa(human, near) < 1.0


def test_confusion_orientation():
    m = confusion([S, S, N], [S, P, N])
    assert m.tolist() == [[1, 1, 0], [0, 0, 0], [0, 0, 1]]


def test_report_fields_and_empty_input():
    rng = np.random.default_rng(0)
    h = [LABELS[i] for i in rng.integers(0, 3, 90)]
    j = [x if rng.random() < 0.7 else LABELS[rng.integers(0, 3)] for x in h]
    r = report(h, j)
    assert r["n"] == 90 and 0 < r["kappa_3class"] < 1 and r["kappa_3class_ci95"][0] < r["kappa_3class"] < r["kappa_3class_ci95"][1]
    with pytest.raises(ValueError):
        kappa([], [])


def test_agreement_script_refuses_blank_labels(tmp_path):
    import csv
    import subprocess
    import sys
    sheet, judge = tmp_path / "s.csv", tmp_path / "j.csv"
    with open(sheet, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "human_label"])
        w.writeheader()
        w.writerows([{"row_id": "a", "human_label": "supported"}, {"row_id": "b", "human_label": ""}])
    with open(judge, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "stratum", "judge_label"])
        w.writeheader()
        w.writerows([{"row_id": "a", "stratum": "system", "judge_label": "supported"},
                     {"row_id": "b", "stratum": "system", "judge_label": "partial"}])
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "scripts/judge_agreement.py"), str(sheet), str(judge), "--out", str(tmp_path / "o.json")],
                       capture_output=True, text=True, env={"PYTHONPATH": str(root / "src"), "PATH": ""})
    assert r.returncode != 0 and "no human label" in (r.stderr + r.stdout)
    assert not (tmp_path / "o.json").exists()
