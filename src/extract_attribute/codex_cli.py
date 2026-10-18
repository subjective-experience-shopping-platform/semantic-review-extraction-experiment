"""기존 실행 옵션으로 리뷰 하나를 Codex CLI에 전달합니다."""

import json
import os
import subprocess
from pathlib import Path

from contracts import parse_units

REASONING_EFFORT = "high"
TIMEOUT_SECONDS = 600


def build_prompt(template: str, review: dict) -> str:
    static, marker, inputs = template.partition("### Input")
    if not marker:
        raise ValueError("프롬프트에 ### Input 구간이 없습니다.")
    values = {
        "product_name": review["productName"],
        "product_category": review["category"],
        "review": review["review"],
    }
    for name, value in values.items():
        placeholder = "{{" + name + "}}"
        if placeholder not in inputs:
            raise ValueError(f"프롬프트에 {placeholder} 입력 변수가 없습니다.")
        inputs = inputs.replace(placeholder, value)
    return static + marker + inputs


def extract_one(
    review: dict,
    task: str,
    template: str,
    schema_path: Path,
    results_dir: Path,
    model: str,
    cli_path: str,
) -> list[dict]:
    result_path = results_dir / f"{review['idx']}.json"
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.pop("CODEX_API_KEY", None)
    command = [
        cli_path,
        "exec",
        "--json",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "--model",
        model,
        "--config",
        f'model_reasoning_effort="{REASONING_EFFORT}"',
        "--config",
        'model_reasoning_summary="none"',
        "--config",
        'model_verbosity="low"',
        "--config",
        'web_search="disabled"',
        "--config",
        "agents.enabled=false",
        "--config",
        'forced_login_method="chatgpt"',
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(result_path),
        "-",
    ]
    completed = subprocess.run(
        command,
        input=build_prompt(template, review),
        text=True,
        encoding="utf-8",
        capture_output=True,
        cwd=results_dir,
        env=env,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{task} review_idx={review['idx']}: Codex CLI 실패\n"
            f"{completed.stdout[-3000:]}\n{completed.stderr[-3000:]}"
        )
    response = json.loads(result_path.read_text(encoding="utf-8"))
    return parse_units(task, review, response)
