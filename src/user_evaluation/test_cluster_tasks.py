"""50:50 선정, 위험 순위, 중복 배제, 대표명 필터와 재현성을 확인합니다."""

import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.user_evaluation.cluster_tasks import build_cluster_tasks
from src.user_evaluation.tasks import build_tasks


def make_diagnostics(size: int = 80) -> dict:
    diagnostics = {}
    for stage, name in (
        ("experiment_b", "attribute"),
        ("experiment_d_aspect", "aspect"),
        ("experiment_d_status", "status"),
    ):
        diagnostics[f"{name}_clusters"] = pd.DataFrame(
            [
                {
                    "stage": stage,
                    "cluster_id": f"{name}-{i:03}",
                    "canonical_label": f"{name}-{i:03}",
                    "medoid_label": f"{name}-{i:03}",
                    "member_count": 2,
                    "member_expressions": [f"{name}-{i:03}", f"{name}-{i:03} 표현"],
                    "aspect_cluster_id": f"aspect-{i % 3:03}" if name == "status" else None,
                    "cluster_max_distance": 0.1 + i / 1000,
                    "cluster_silhouette_mean": 0.2,
                }
                for i in range(size)
            ]
        )
    return diagnostics


@pytest.mark.parametrize(
    ("count", "risky_count", "random_count"),
    [(100, 50, 50), (10, 5, 5), (7, 3, 4), (4, 2, 2), (1, 0, 1)],
)
def test_ratios_unique_clusters_and_roundtrip(tmp_path, count, risky_count, random_count):
    diagnostics = make_diagnostics()
    frame = pd.DataFrame(
        {
            "idx": range(count),
            "review_idx": range(count),
            "attribute": "화질",
            "aspect": "화질",
            "status": "선명함",
        }
    )
    tasks = build_tasks(
        {f"experiment_{name}": frame for name in "abcd"},
        diagnostics,
        {"task_count": count, "random_seed": 7},
    )
    clusters = tasks["cluster_tasks"]
    assert len(tasks["review_tasks"]) == len(clusters) == count
    assert clusters["selection_reason"].value_counts().to_dict() == {
        key: value for key, value in {"risky": risky_count, "random": random_count}.items() if value
    }
    assert clusters["cluster_id"].is_unique
    assert clusters["cluster_validation_id"].is_unique
    assert clusters["canonical_label"].notna().all()
    assert "task_type" not in clusters
    assert not any(column.startswith("paired_") or column == "pair_distance" for column in clusters)
    assert_frame_equal(clusters, build_cluster_tasks(diagnostics, count=count, seed=7))
    clusters.to_parquet(tmp_path / "tasks.parquet", index=False)
    assert_frame_equal(clusters, pd.read_parquet(tmp_path / "tasks.parquet"))
    parents = diagnostics["aspect_clusters"].set_index("cluster_id")["canonical_label"]
    for row in clusters.itertuples(index=False):
        assert len(json.loads(row.members)) >= 2
        if row.stage == "experiment_d_status":
            source = diagnostics["status_clusters"].set_index("cluster_id").loc[row.cluster_id]
            assert row.parent_aspect_label == parents[source.aspect_cluster_id]


def test_priority_ranking_is_balanced_and_random_uses_remaining_clusters():
    diagnostics = make_diagnostics(size=8)
    tasks = build_cluster_tasks(diagnostics, count=10, seed=7)
    risky = tasks[tasks.selection_reason.eq("risky")]
    random = tasks[tasks.selection_reason.eq("random")]
    assert set(risky.cluster_id) == {
        "attribute-007",
        "aspect-007",
        "status-007",
        "attribute-006",
        "aspect-006",
    }
    assert not set(risky.cluster_id) & set(random.cluster_id)
    assert risky.stage.value_counts().max() - risky.stage.value_counts().min() <= 1
    source = pd.concat(diagnostics.values()).set_index("cluster_id")
    for row in tasks.itertuples(index=False):
        assert row.risk_score == pytest.approx(
            source.at[row.cluster_id, "cluster_max_distance"] + 0.4
        )
    other = build_cluster_tasks(diagnostics, count=10, seed=17)
    assert set(other.loc[other.selection_reason.eq("risky"), "cluster_id"]) == set(risky.cluster_id)
    assert set(other.loc[other.selection_reason.eq("random"), "cluster_id"]) != set(
        random.cluster_id
    )


def test_unnamed_and_singleton_clusters_are_excluded_and_missing_silhouette_is_supported():
    diagnostics = make_diagnostics()
    frame = diagnostics["attribute_clusters"]
    frame.loc[[0, 1], "cluster_max_distance"] = 2.0
    frame.at[0, "canonical_label"] = None
    frame.at[1, "member_count"] = 1
    for clusters in diagnostics.values():
        clusters["cluster_silhouette_mean"] = None
    tasks = build_cluster_tasks(diagnostics, count=100, seed=7)
    assert not tasks.cluster_id.isin(["attribute-000", "attribute-001"]).any()
    source = pd.concat(diagnostics.values()).set_index("cluster_id")
    for row in tasks.itertuples(index=False):
        assert row.risk_score == pytest.approx(
            source.at[row.cluster_id, "cluster_max_distance"] + 0.5
        )


def test_too_few_eligible_clusters_fail_before_sampling():
    diagnostics = make_diagnostics(size=4)
    for frame in diagnostics.values():
        frame.loc[[0, 1], "canonical_label"] = None
    with pytest.raises(ValueError, match="후보가 부족"):
        build_cluster_tasks(diagnostics, count=10, seed=7)


@pytest.mark.parametrize("count", [0, -1])
def test_nonpositive_task_counts_are_rejected(count):
    with pytest.raises(ValueError, match="1 이상"):
        build_cluster_tasks(make_diagnostics(), count=count, seed=7)
