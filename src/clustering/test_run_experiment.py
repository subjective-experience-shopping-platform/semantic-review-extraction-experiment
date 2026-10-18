"""설정된 units 입력 경로와 실험별 보존 사본을 확인합니다."""

import json

import pandas as pd
from pandas.testing import assert_frame_equal

import run_experiment as runner
from src.report.build_report import ReportResult


def test_configured_units_are_used_and_copied_to_experiment(tmp_path, monkeypatch):
    paths = [
        tmp_path / "inputs/reviews.parquet",
        tmp_path / "inputs/chosen_attributes.parquet",
        tmp_path / "other/chosen_opinions.parquet",
    ]
    frames = [
        pd.DataFrame(
            [
                {
                    "idx": 1,
                    "review": "화면이 선명합니다.",
                    "productName": "모니터",
                    "category": "모니터",
                }
            ]
        ),
        pd.DataFrame(
            [{"idx": 1, "review_idx": 1, "raw_attribute": "화질", "sentiment": "positive"}]
        ),
        pd.DataFrame(
            [
                {
                    "idx": 1,
                    "review_idx": 1,
                    "raw_aspect": "화질",
                    "raw_status": None,
                    "excerpt": "화면이 선명합니다.",
                    "opinion": "화면이 선명함",
                    "sentiment": "positive",
                }
            ]
        ),
    ]
    for path, frame in zip(paths, frames, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    original = {path: path.read_bytes() for path in paths}
    config = {
        "origin_dataset": {"output_path": "inputs/reviews.parquet"},
        "extract_attribute": {
            "attribute_units": "inputs/chosen_attributes.parquet",
            "opinion_units": str(paths[2]),
        },
        "experiment": {"output_dir": "experiment"},
        "embedding": {},
        "clustering": {},
        "user_evaluation": {"task_count": 1, "random_seed": 7},
    }
    (tmp_path / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "SentenceEncoder", lambda *_: object())
    report_calls = []

    def fake_report(directory):
        assert (directory / "data/dataset.parquet").is_file()
        assert (directory / "user_evaluation/review_tasks.parquet").is_file()
        report_calls.append(directory)
        return ReportResult(directory / "report.md", 0)

    monkeypatch.setattr(runner, "build_report", fake_report)

    def fake_experiments(attribute, opinion, encoder, settings):
        assert_frame_equal(attribute, pd.read_parquet(paths[1]))
        assert_frame_equal(opinion, pd.read_parquet(paths[2]))
        return {
            "experiment_a": attribute,
            "experiment_b": attribute,
            "experiment_c": opinion.assign(status=opinion["raw_status"]),
            "experiment_d": opinion.assign(status=opinion["raw_status"]),
        }, {}

    monkeypatch.setattr(runner, "build_experiments", fake_experiments)
    monkeypatch.setattr(
        runner,
        "build_tasks",
        lambda *_: {
            "review_tasks": pd.DataFrame({"review_idx": [1]}),
            "cluster_tasks": pd.DataFrame({"selection_reason": ["random"]}),
        },
    )
    runner.main()
    experiments = list((tmp_path / "experiment").iterdir())
    assert len(experiments) == 1
    assert report_calls == experiments
    for source, name in zip(
        paths, ("dataset.parquet", "attribute_units.parquet", "opinion_units.parquet"), strict=True
    ):
        assert (experiments[0] / "data" / name).read_bytes() == original[source]
        assert source.read_bytes() == original[source]
    assert not (tmp_path / "data/attribute_units.parquet").exists()
    assert not (tmp_path / "data/opinion_units.parquet").exists()
