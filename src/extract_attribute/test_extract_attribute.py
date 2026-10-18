"""원문 근거, 빈 결과, 저장 및 임시 폴더 정리 동작을 확인합니다."""

import json
import subprocess

import pandas as pd
import pytest
from codex_cli import build_prompt, extract_one
from contracts import parse_units

import extract_attribute as runner


def test_invalid_unit_does_not_remove_valid_units():
    review = {"idx": 12, "review": "화면이 선명합니다."}
    valid = {
        "raw_aspect": "화면 선명도",
        "raw_status": None,
        "excerpt": "화면이 선명합니다.",
        "opinion": "화면이 선명함",
        "sentiment": "positive",
    }
    invalid = {**valid, "excerpt": "배송이 빠릅니다."}
    units = parse_units("opinion_units", review, {"opinion_units": [invalid, valid]})
    assert len(units) == 1 and units[0]["raw_status"] is None
    assert units[0]["review_idx"] == 12


def test_text_cleanup_and_schema():
    review = {"idx": 12, "review": "화면이 선명합니다."}
    result = parse_units(
        "attribute_units",
        review,
        {"attribute_units": [{"raw_attribute": " 화면\t선명도\n", "sentiment": "positive"}]},
    )
    assert result[0]["raw_attribute"] == "화면 선명도"
    assert parse_units("attribute_units", review, {"attribute_units": []}) == []
    with pytest.raises(ValueError):
        parse_units("attribute_units", review, {"representative_attributes": []})


def test_prompt_preserves_static_json():
    template = (
        '{"attribute_units": []}\n### Input\n{{product_name}} {{product_category}} {{review}}'
    )
    prompt = build_prompt(
        template,
        {
            "productName": "모니터A",
            "category": "모니터",
            "review": "선명함 {원문}",
        },
    )
    assert prompt == '{"attribute_units": []}\n### Input\n모니터A 모니터 선명함 {원문}'


def test_cli_uses_configured_executable_and_model(tmp_path, monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        result = command[command.index("--output-last-message") + 1]
        with open(result, "w", encoding="utf-8") as file:
            json.dump({"attribute_units": []}, file)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("codex_cli.subprocess.run", fake_run)
    review = {"idx": 12, "productName": "모니터A", "category": "모니터", "review": "택배 도착"}
    template = "### Input\n{{product_name}} {{product_category}} {{review}}"
    result = extract_one(
        review,
        "attribute_units",
        template,
        tmp_path / "schema.json",
        tmp_path,
        "gpt-6-luna",
        "/configured/codex",
    )
    assert result == []
    command = captured["command"]
    assert command[0] == "/configured/codex"
    assert command[command.index("--model") + 1] == "gpt-6-luna"


@pytest.mark.parametrize("absolute", [False, True])
def test_empty_results_saved_and_temp_removed_only_after_success(tmp_path, monkeypatch, absolute):
    (tmp_path / "data").mkdir()
    output_paths = {
        task: tmp_path / "custom" / f"selected_{task}.parquet" for task in runner.FIELDS
    }
    config = {
        "origin_dataset": {"output_path": "data/input.parquet"},
        "extract_attribute": {
            "model": "gpt-6-luna",
            "cli_path": "/test/codex",
            **{
                task: str(path if absolute else path.relative_to(tmp_path))
                for task, path in output_paths.items()
            },
        },
    }
    (tmp_path / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "idx": 12,
                "review": "택배가 빨리 왔습니다.",
                "productName": "모니터A",
                "category": "모니터",
            }
        ]
    ).to_parquet(tmp_path / "data/input.parquet", index=False)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    def fake_task(task, reviews, temporary_dir, model, cli_path, max_workers):
        assert model == "gpt-6-luna"
        assert cli_path == "/test/codex"
        (temporary_dir / f"{task}.json").write_text("{}", encoding="utf-8")
        columns = ["idx", "review_idx", *runner.FIELDS[task]]
        return pd.DataFrame(columns=columns)

    monkeypatch.setattr(runner, "extract_task", fake_task)
    runner.main()
    for task, path in output_paths.items():
        assert pd.read_parquet(path).empty
        assert not (tmp_path / f"data/{task}.parquet").exists()
    assert not list((tmp_path / "data").glob(".extract_attribute-*"))
    original_outputs = {path: path.read_bytes() for path in output_paths.values()}

    def fail_task(task, reviews, temporary_dir, model, cli_path, max_workers):
        (temporary_dir / "completed.json").write_text("{}", encoding="utf-8")
        raise RuntimeError("테스트용 실패")

    monkeypatch.setattr(runner, "extract_task", fail_task)
    with pytest.raises(RuntimeError, match="테스트용 실패"):
        runner.main()
    assert list((tmp_path / "data").glob(".extract_attribute-*/completed.json"))
    assert all(p.read_bytes() == contents for p, contents in original_outputs.items())


@pytest.mark.parametrize("attribute_path", ["data/same.parquet", "data/input.parquet"])
def test_conflicting_output_paths_are_rejected_before_extraction(
    tmp_path, monkeypatch, attribute_path
):
    config = {
        "origin_dataset": {"output_path": "data/input.parquet"},
        "extract_attribute": {
            "attribute_units": attribute_path,
            "opinion_units": "data/same.parquet",
        },
    }
    (tmp_path / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="겹칠 수 없습니다"):
        runner.main()
