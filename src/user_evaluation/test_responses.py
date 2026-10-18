"""선정 방식과 무관한 동일 지표 저장과 불완전 응답 거부를 확인합니다."""

import pandas as pd
import pytest

from src.user_evaluation.responses import REVIEW_SCORES, save_responses


def test_uniform_cluster_responses_and_required_answers(tmp_path):
    reviews = [{"review_idx": 1}]
    review_rows = [
        {
            "review_idx": 1,
            "experiment": name,
            **dict.fromkeys(REVIEW_SCORES, 4),
            "preferred": name == "D",
        }
        for name in "ABCD"
    ]
    tasks = [
        {
            "cluster_validation_id": "task-1",
            "selection_reason": "risky",
            "stage": "experiment_b",
            "cluster_id": "B-1",
            "canonical_label": "화질",
        },
        {
            "cluster_validation_id": "task-2",
            "selection_reason": "random",
            "stage": "experiment_b",
            "cluster_id": "B-2",
            "canonical_label": "선명도",
        },
    ]
    rows = [
        {
            "cluster_validation_id": "task-1",
            "cluster_id": "B-1",
            "cohesion": 5,
            "canonical_label_fit": 4,
        },
        {
            "cluster_validation_id": "task-2",
            "cluster_id": "B-2",
            "cohesion": 3,
            "canonical_label_fit": 2,
        },
    ]
    for field, index, invalid in (
        ("cohesion", 0, None),
        ("canonical_label_fit", 1, None),
        ("cohesion", 0, 6),
        ("cluster_id", 0, "other"),
    ):
        wrong = [row.copy() for row in rows]
        wrong[index][field] = invalid
        with pytest.raises(ValueError):
            save_responses("invalid", review_rows, wrong, reviews, tasks, tmp_path, "signature")
        assert not (tmp_path / "invalid.parquet").exists()
    for incomplete in (rows[:1], rows + [rows[0]], [{**rows[0]}, {**rows[0]}]):
        with pytest.raises(ValueError):
            save_responses(
                "invalid", review_rows, incomplete, reviews, tasks, tmp_path, "signature"
            )
    for unsupported in ({**tasks[0], "task_type": "pair"}, {**tasks[0], "canonical_label": None}):
        with pytest.raises(ValueError, match="새 정책"):
            save_responses(
                "invalid",
                review_rows,
                rows,
                reviews,
                [unsupported, tasks[1]],
                tmp_path,
                "signature",
            )
    for preferred in ([], ["A", "D"], list("ABCD")):
        invalid = [{**row, "preferred": row["experiment"] in preferred} for row in review_rows]
        with pytest.raises(ValueError, match="정확히 하나"):
            save_responses("invalid", invalid, rows, reviews, tasks, tmp_path, "signature")
        assert not (tmp_path / "invalid.parquet").exists()
    for invalid_value in (None, 1, "true"):
        invalid = [dict(row) for row in review_rows]
        invalid[0]["preferred"] = invalid_value
        with pytest.raises(ValueError, match="boolean"):
            save_responses("invalid", invalid, rows, reviews, tasks, tmp_path, "signature")
        assert not (tmp_path / "invalid.parquet").exists()
    path = save_responses("valid", review_rows, rows, reviews, tasks, tmp_path, "signature")
    saved = pd.read_parquet(path)
    assert len(saved) == 6
    assert saved.loc[saved.evaluation_level.eq("review"), "preferred"].sum() == 1
    assert not {"paired_cluster_id", "task_type", "merge_decision"} & set(saved.columns)
    clusters = saved.loc[saved["evaluation_level"].eq("cluster")].set_index("cluster_id")
    assert clusters["cohesion"].tolist() == [5, 3]
    assert clusters["canonical_label_fit"].tolist() == [4, 2]
    assert clusters["selection_reason"].tolist() == ["risky", "random"]
