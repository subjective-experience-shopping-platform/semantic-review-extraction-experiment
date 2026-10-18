"""실험 선택과 선택된 실험 내부의 완료 응답 저장을 검증합니다."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from src.user_evaluation.experiments import list_experiments


def make_experiment(root: Path, name: str, label: str) -> Path:
    directory = root / "experiment" / name
    clustering = directory / "clustering"
    evaluation = directory / "user_evaluation"
    clustering.mkdir(parents=True)
    evaluation.mkdir()
    data = directory / "data"
    data.mkdir()
    pd.DataFrame(
        [
            {
                "idx": 1,
                "review": f"{label}가 선명해서 좋습니다.",
                "productName": "실험 당시 모니터",
            }
        ]
    ).to_parquet(data / "dataset.parquet", index=False)
    units = pd.DataFrame({"idx": [1, 2], "review_idx": [1, 1]})
    units.to_parquet(data / "attribute_units.parquet", index=False)
    units.to_parquet(data / "opinion_units.parquet", index=False)
    for experiment in "abcd":
        units.to_parquet(clustering / f"experiment_{experiment}.parquet", index=False)
    for name in ("attribute", "aspect", "status"):
        pd.DataFrame({"cluster_id": ["group", "group"]}).to_parquet(
            clustering / f"{name}_nodes.parquet", index=False
        )
        pd.DataFrame(
            {"member_count": [2], "source_row_count": [2], "cluster_silhouette_mean": [None]}
        ).to_parquet(clustering / f"{name}_clusters.parquet", index=False)
    tasks = {"review_idx": [1]}
    tasks.update({f"result_{experiment}": [json.dumps([label])] for experiment in "abcd"})
    pd.DataFrame(tasks).to_parquet(evaluation / "review_tasks.parquet", index=False)
    pd.DataFrame(
        [
            {
                "cluster_validation_id": "experiment_b:B-1",
                "stage": "experiment_b",
                "cluster_id": "B-1",
                "parent_aspect_label": None,
                "canonical_label": label,
                "members": json.dumps([label, f"화면 {label}"]),
            }
        ]
    ).to_parquet(evaluation / "cluster_tasks.parquet", index=False)
    return directory


def test_list_only_complete_experiments_and_count_completed_files(tmp_path):
    first = make_experiment(tmp_path, "2026-10-05_090000", "화질")
    make_experiment(tmp_path, "2026-10-05_100000", "밝기")
    completed = first / "user_evaluation/completed_evaluators"
    completed.mkdir()
    (completed / "one.parquet").touch()
    (completed / "two.parquet").touch()
    (completed / ".pending.parquet").touch()
    (completed / "note.txt").touch()
    (tmp_path / "experiment/2026-10-05_110000").mkdir()
    (tmp_path / "experiment/.clustering-pending").mkdir()
    experiments = list_experiments(tmp_path / "experiment")
    assert experiments == [
        {"name": "2026-10-05_100000", "label": "2026-10-05 100000", "completed_count": 0},
        {"name": "2026-10-05_090000", "label": "2026-10-05 090000", "completed_count": 2},
    ]


@pytest.mark.parametrize("cluster_case", ["random", "risky"])
def test_selection_then_complete_and_save_to_selected_experiment(tmp_path, cluster_case):
    first = make_experiment(tmp_path, "2026-10-05_090000", "화질")
    second = make_experiment(tmp_path, "2026-10-05_100000", "밝기")
    cluster_path = first / "user_evaluation/cluster_tasks.parquet"
    tasks = pd.read_parquet(cluster_path)
    tasks["selection_reason"] = cluster_case
    tasks.to_parquet(cluster_path, index=False)
    (tmp_path / "data").mkdir()
    pd.DataFrame(
        [{"idx": 1, "review": "실험 이후 바뀐 원문입니다.", "productName": "현재 모니터"}]
    ).to_parquet(
        tmp_path / "data/dataset.parquet",
        index=False,
    )
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": {"output_dir": "experiment"},
                "origin_dataset": {"output_path": "data/dataset.parquet"},
            }
        ),
        encoding="utf-8",
    )
    app = AppTest.from_string(
        "from pathlib import Path\nfrom src.user_evaluation import app\n"
        f"app.PROJECT_ROOT = Path({str(tmp_path)!r})\napp.main()\n"
    ).run()
    assert not app.exception
    assert not app.radio
    assert app.selectbox[0].options == [
        "2026-10-05 100000 · 완료 0명",
        "2026-10-05 090000 · 완료 0명",
    ]
    app.selectbox[0].set_value("2026-10-05_090000").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["selected_experiment"] == first.name
    score_keys = [
        button.key for button in app.button if button.key and button.key.startswith("score_button_")
    ]
    assert len(score_keys) == 60
    # 단축키는 A–D 내부 순서가 아닌 화면의 결과 1~4 순서로 연결합니다.
    visual_order = [
        key.removeprefix("score_button_").removesuffix("_1")
        for key in score_keys
        if key.endswith("_1")
    ]
    assert app.session_state["keyboard_score_keys"] == visual_order
    assert app.session_state["keyboard_score_index"] == 0
    assert all(not button.proto.help for button in app.button)
    assert any("화질가 선명해서 좋습니다." in element.value for element in app.markdown)
    assert not any("실험 이후 바뀐 원문입니다." in element.value for element in app.markdown)
    # 동일한 결과의 최초 점수 동기화와 이후 독립 수정이 유지되는지 확인합니다.
    first_key = "score_button_score::1::A::factual_faithfulness_4"
    app.button(key=first_key).click().run()
    assert app.session_state["score::1::B::factual_faithfulness"] == 4
    app.button(key="score_button_score::1::B::factual_faithfulness_2").click().run()
    assert app.session_state["score::1::A::factual_faithfulness"] == 4
    assert app.session_state["score::1::B::factual_faithfulness"] == 2
    if cluster_case == "random":
        app.session_state["keyboard_score_index"] = 0
        app.run()
        for index in range(13):
            score = index % 5 + 1
            active = [button for button in app.button if button.proto.shortcut == str(score)]
            assert len(active) == 1
            expected_key = visual_order[index % 12]
            assert active[0].key == f"score_button_{expected_key}_{score}"
            active[0].click().run()
            assert not app.exception
            assert app.session_state[expected_key] == score
            assert app.session_state["keyboard_score_index"] == (index + 1) % 12
    for key in score_keys:
        if key.endswith("_4"):
            app.button(key=key).click().run()
    assert not app.checkbox
    assert app.radio[0].value is None
    assert app.radio[0].options == ["결과 1", "결과 2", "결과 3", "결과 4"]
    assert next(button for button in app.button if button.label == "군집 평가로 이동").disabled
    app.radio[0].set_value("A").run()
    app.radio[0].set_value("D").run()
    assert app.radio[0].value == "D"
    next(button for button in app.button if button.label == "군집 평가로 이동").click().run()
    assert not app.exception
    cluster_score_keys = [
        button.key for button in app.button if button.key and button.key.startswith("score_button_")
    ]
    assert len(cluster_score_keys) == 10
    assert app.session_state["keyboard_score_index"] == 0
    cluster_order = [
        key.removeprefix("score_button_").removesuffix("_1")
        for key in cluster_score_keys
        if key.endswith("_1")
    ]
    assert app.session_state["keyboard_score_keys"] == cluster_order
    # 모든 군집 문항의 두 지표를 순회합니다.
    for index in range(len(cluster_order) + 1):
        active = [button for button in app.button if button.proto.shortcut == "3"]
        assert len(active) == 1
        expected_key = cluster_order[index % len(cluster_order)]
        assert active[0].key == f"score_button_{expected_key}_3"
        active[0].click().run()
        assert app.session_state[expected_key] == 3
        assert app.session_state["keyboard_score_index"] == (index + 1) % len(cluster_order)
    # 저장 가능 상태 검사는 이전 선택을 지우고 시작합니다.
    for key in cluster_order:
        app.session_state[key] = None
    app.run()
    assert next(button for button in app.button if button.label == "평가 완료 및 저장").disabled
    assert not any("합쳐도" in element.value for element in app.markdown)
    assert all(not button.proto.help for button in app.button)
    for key in cluster_score_keys:
        if key.endswith("_5"):
            app.button(key=key).click().run()
    next(button for button in app.button if button.label == "평가 완료 및 저장").click().run()
    assert not app.exception
    assert app.success
    assert not app.error
    assert any("2명의 평가가 더 필요" in element.value for element in app.info)
    assert (first / "report.md").read_text().endswith("<사용자 평가 필요.(1명/3명)>\n")
    files = list((first / "user_evaluation/completed_evaluators").glob("*.parquet"))
    assert len(files) == 1
    saved = pd.read_parquet(files[0])
    assert len(saved) == 5
    preferred = saved.loc[saved.evaluation_level.eq("review") & saved.preferred.fillna(False)]
    assert preferred.experiment.tolist() == ["D"]
    cluster = saved.loc[saved["evaluation_level"].eq("cluster")].iloc[0]
    assert cluster["cluster_id"] == "B-1"
    assert cluster["cohesion"] == 5
    assert cluster["canonical_label_fit"] == 5
    assert cluster["selection_reason"] == cluster_case
    assert not {"paired_cluster_id", "task_type", "merge_decision"} & set(saved.columns)
    assert not (second / "user_evaluation/completed_evaluators").exists()
    next(button for button in app.button if button.label == "실험 선택으로 돌아가기").click().run()
    assert not app.exception
    assert not app.radio
    assert "2026-10-05 090000 · 완료 1명" in app.selectbox[0].options
    assert "review_responses" not in app.session_state
    assert "keyboard_score_keys" not in app.session_state


@pytest.mark.parametrize("invalid", ["pair", "unnamed"])
def test_old_pair_and_unnamed_questions_require_new_experiment(tmp_path, invalid):
    directory = make_experiment(tmp_path, "2026-10-05_090000", "화질")
    path = directory / "user_evaluation/cluster_tasks.parquet"
    tasks = pd.read_parquet(path)
    if invalid == "pair":
        tasks["task_type"] = "pair"
    else:
        tasks["canonical_label"] = None
    tasks.to_parquet(path, index=False)
    (tmp_path / "config.yaml").write_text("experiment:\n  output_dir: experiment\n")
    app = AppTest.from_string(
        "from pathlib import Path\nfrom src.user_evaluation import app\n"
        f"app.PROJECT_ROOT = Path({str(tmp_path)!r})\napp.main()\n"
    ).run()
    app.button[0].click().run()
    assert not app.exception
    assert app.error
    assert "문항입니다" in app.error[0].value
    assert not (directory / "user_evaluation/completed_evaluators").exists()
