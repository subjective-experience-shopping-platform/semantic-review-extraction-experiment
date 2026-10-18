"""고정 의미 규칙과 사용자 평가에 쓰이는 군집 진단값을 검증합니다."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.clustering.clustering import cluster_expressions, opposite_status

SETTINGS = yaml.safe_load((Path(__file__).resolve().parents[2] / "config.yaml").read_text())[
    "clustering"
]


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("빠름", "느림"),
        ("밝음", "어두움"),
        ("소음 있음", "소음 없음"),
        ("문제 발생", "문제 미발생"),
        ("선명함", "선명함 아님"),
        ("작동", "안 작동"),
    ],
)
def test_opposites_without_configured_rule_lists(left, right):
    assert opposite_status(left, right, SETTINGS)
    assert opposite_status(right, left, SETTINGS)
    assert not opposite_status("선명함", "밝음", SETTINGS)


def test_fixed_role_filter_and_diagnostic_values():
    class Encoder:
        def encode(self, texts):
            vectors = {"좋음": [1, 0], "화질": [0.99, 0.01], "무게": [0, 1]}
            return np.array([vectors[text] for text in texts], dtype=float)

    frame = pd.DataFrame(
        {"idx": [1, 2, 3], "review_idx": [1, 2, 3], "text": ["좋음", "화질", "무게"]}
    )
    nodes, clusters = cluster_expressions(
        frame, "text", Encoder(), SETTINGS, 0.22, "experiment_b", "B", "aspect"
    )
    multi = clusters.loc[clusters["member_count"].gt(1)].iloc[0]
    assert multi["canonical_label"] == "화질"
    assert 0.9 < multi["cluster_silhouette_mean"] <= 1
    assert "centroid_embedding" not in multi.index
    assert nodes["cluster_id"].notna().all()
    assert not {
        key for key in SETTINGS if "disallowed" in key or "opposition" in key or "negation" in key
    }
