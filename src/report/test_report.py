"""3명 조건, 집계 분모, 계산 불가, 재실행과 실패 시 기존 보고서 보존을 확인합니다."""

import hashlib

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.report.build_report import build_report
from src.report.user_results import preference_intervals
from src.user_evaluation import app as evaluation_app
from src.user_evaluation.responses import REVIEW_SCORES, save_responses


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / "2026-10-05_111111"
    for name in ("data", "clustering", "user_evaluation"):
        (root / name).mkdir(parents=True)
    pd.DataFrame({"idx": [1, 2, 3]}).to_parquet(root / "data/dataset.parquet", index=False)
    units = pd.DataFrame({"idx": [1, 2, 3, 4], "review_idx": [1, 1, 2, 2]})
    for name in ("attribute_units", "opinion_units"):
        units.to_parquet(root / f"data/{name}.parquet", index=False)
    units.to_parquet(root / "clustering/experiment_c.parquet", index=False)
    ids = {"attribute": "B-1", "aspect": "D-A-1", "status": "D-S-1"}
    tasks = []
    for name, cluster_id in ids.items():
        nodes = pd.DataFrame({"cluster_id": [cluster_id] * 2 + [cluster_id + "-single"]})
        clusters = pd.DataFrame(
            {
                "cluster_id": [cluster_id, cluster_id + "-single"],
                "member_count": [2, 1],
                "source_row_count": [3, 1],
                "cluster_silhouette_mean": [0.6, 0.0] if name != "status" else [None, None],
            }
        )
        nodes.to_parquet(root / f"clustering/{name}_nodes.parquet", index=False)
        clusters.to_parquet(root / f"clustering/{name}_clusters.parquet", index=False)
        stage = "experiment_b" if name == "attribute" else f"experiment_d_{name}"
        tasks.append(
            {
                "cluster_validation_id": f"{stage}:{cluster_id}",
                "cluster_id": cluster_id,
                "stage": stage,
                "canonical_label": "화질",
                "selection_reason": "risky",
            }
        )
    pd.DataFrame(tasks).to_parquet(root / "user_evaluation/cluster_tasks.parquet", index=False)
    pd.DataFrame({"review_idx": [1, 2]}).to_parquet(
        root / "user_evaluation/review_tasks.parquet", index=False
    )
    return root


def complete(root, number):
    review_tasks = pd.read_parquet(root / "user_evaluation/review_tasks.parquet").to_dict("records")
    cluster_tasks = pd.read_parquet(root / "user_evaluation/cluster_tasks.parquet").to_dict(
        "records"
    )
    review = [
        {
            "review_idx": task["review_idx"],
            "experiment": name,
            **dict.fromkeys(REVIEW_SCORES, index + 1),
            "preferred": name == ("A" if task["review_idx"] == 1 and number == 1 else "D"),
        }
        for task in review_tasks
        for index, name in enumerate("ABCD")
    ]
    cluster = [
        {
            "cluster_validation_id": task["cluster_validation_id"],
            "cluster_id": task["cluster_id"],
            "cohesion": [1, 3, 5][number - 1],
            "canonical_label_fit": [2, 4, 5][number - 1],
        }
        for task in cluster_tasks
    ]
    signature = hashlib.sha256()
    for path in (
        root / "data/dataset.parquet",
        root / "user_evaluation/review_tasks.parquet",
        root / "user_evaluation/cluster_tasks.parquet",
    ):
        signature.update(path.read_bytes())
    return save_responses(
        f"reviewer-{number}",
        review,
        cluster,
        review_tasks,
        cluster_tasks,
        root / "user_evaluation/completed_evaluators",
        signature.hexdigest(),
    )


def test_threshold_idempotence_and_no_analysis_outputs(experiment):
    originals = {p: p.read_bytes() for p in experiment.rglob("*.parquet")}
    result = build_report(experiment)
    static = result.path.read_text().split("## 사용자 평가")[0]
    assert "| B attribute | 3 | 2 | 33.33% | 1(50.0%) | 1(25.0%) | 0.400 |" in static
    assert "| D status | 3 | 2 | 33.33% | 1(50.0%) | 1(25.0%) | 계산 불가 |" in static
    for count in range(4):
        if count:
            complete(experiment, count)
        result = build_report(experiment)
        assert result.completed_count == count
        assert result.remaining_count == 3 - count
        content = result.path.read_text()
        assert content.split("## 사용자 평가")[0] == static
        if count < 3:
            assert content.endswith(f"<사용자 평가 필요.({count}명/3명)>\n")
        else:
            assert "사용자 평가 필요" not in content
            assert "선택 수의 총합은 6" in content
            assert "| A | 1 | 16.67% |" in content
            assert "| D | 5 | 83.33% |" in content
            assert "| 세 기준 단순평균 | 1.0000 | 2.0000 | 3.0000 | 4.0000 |" in content
            assert "| B-속성 | 응집성 | 3 | 3.0000 | 2.0000 | 3 | 1 / 0 / 1 |" in content
            assert "총 6개" in content
            assert "{" not in content
        assert build_report(experiment).path.read_text() == content
    assert all(path.read_bytes() == value for path, value in originals.items())
    assert set(experiment.rglob("*")) == {
        *originals,
        experiment / "data",
        experiment / "clustering",
        experiment / "user_evaluation",
        experiment / "report.md",
        experiment / "user_evaluation/completed_evaluators",
        *(
            experiment / f"user_evaluation/completed_evaluators/reviewer-{i}.parquet"
            for i in (1, 2, 3)
        ),
    }


@pytest.mark.parametrize("invalid", ["signature", "incomplete", "preferred", "multiple", "none"])
def test_invalid_completion_keeps_previous_report(experiment, invalid):
    result = build_report(experiment)
    before = result.path.read_bytes()
    path = complete(experiment, 1)
    frame = pd.read_parquet(path)
    if invalid == "signature":
        frame["task_signature"] = "old-tasks"
    elif invalid == "incomplete":
        frame = frame.iloc[:-1]
    elif invalid == "preferred":
        frame["preferred"] = frame["preferred"].astype("Int64")
    else:
        review = frame.evaluation_level.eq("review")
        frame.loc[review, "preferred"] = invalid == "multiple"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="완료 응답"):
        build_report(experiment)
    assert result.path.read_bytes() == before
    assert not list(experiment.glob(".report*"))


def test_all_null_status_does_not_invent_rates(experiment):
    for name in ("nodes", "clusters"):
        path = experiment / f"clustering/status_{name}.parquet"
        pd.read_parquet(path).iloc[:0].to_parquet(path, index=False)
    content = build_report(experiment).path.read_text()
    assert "| D status | 0 | 0 | 계산 불가 | 0(계산 불가) | 0(계산 불가) | 계산 불가 |" in content


def test_bootstrap_keeps_same_review_votes_together():
    # 각 리뷰의 득표 구성이 같으면 묶음 재표집의 구간은 점 추정값과 같습니다.
    review = pd.DataFrame(
        [
            {"review_idx": idx, "experiment": name, "preferred": name == preferred}
            for idx in (1, 2)
            for preferred in ("C", "D", "D")
            for name in "ABCD"
        ]
    )
    intervals = preference_intervals(review)
    assert intervals[0].tolist() == pytest.approx([0, 0, 100 / 3, 200 / 3])
    assert intervals[1].tolist() == pytest.approx(intervals[0].tolist())


@pytest.mark.parametrize("count", [1, 2])
def test_completion_screen_shows_remaining_evaluators(experiment, count):
    for number in range(1, count + 1):
        complete(experiment, number)
    screen = AppTest.from_string(
        "from pathlib import Path\nfrom src.user_evaluation.app import render_completion\n"
        f"render_completion(Path({str(experiment)!r}))\n"
    ).run()
    assert not screen.exception
    assert not screen.error
    assert any(f"{3 - count}명의 평가가 더 필요" in element.value for element in screen.info)
    assert screen.session_state["report_result"].completed_count == count
    assert (experiment / "report.md").read_text().endswith(f"<사용자 평가 필요.({count}명/3명)>\n")


def test_report_retry_does_not_resave_evaluation(experiment, monkeypatch):
    response = complete(experiment, 1)
    original = response.read_bytes()
    calls = []

    def fail_once(directory):
        calls.append(directory)
        if len(calls) == 1:
            raise RuntimeError("보고서 저장 실패 예시")
        return build_report(directory)

    monkeypatch.setattr(evaluation_app, "build_report", fail_once)
    screen = AppTest.from_string(
        "from pathlib import Path\nfrom src.user_evaluation.app import render_completion\n"
        f"render_completion(Path({str(experiment)!r}))\n"
    ).run()
    assert not screen.exception
    assert screen.error
    assert any(
        "src/report/build_report.py --experiment" in element.value for element in screen.code
    )
    next(button for button in screen.button if button.label == "보고서 생성 재시도").click().run()
    assert not screen.exception
    assert not screen.error
    assert len(calls) == 2
    assert screen.session_state["report_result"].completed_count == 1
    assert response.read_bytes() == original
    assert len(list(response.parent.glob("*.parquet"))) == 1
    screen.run()
    assert len(calls) == 2
