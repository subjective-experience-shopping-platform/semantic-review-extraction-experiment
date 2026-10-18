"""실험 실행, 평가 완료 화면, 수동 복구 CLI가 공유하는 보고서 진입점입니다."""

import argparse
import fcntl
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.report.static_results import static_section
from src.report.user_results import load_completed, user_section

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_EVALUATORS = 3


@dataclass(frozen=True)
class ReportResult:
    path: Path
    completed_count: int

    @property
    def remaining_count(self) -> int:
        return max(0, REQUIRED_EVALUATORS - self.completed_count)


def build_report(experiment_dir: Path) -> ReportResult:
    """완료 파일을 기준으로 보고서를 갱신합니다. 재실행은 인원을 늘리지 않습니다."""
    directory = Path(experiment_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"실험 폴더가 없습니다: {directory}")
    name = directory.name
    timestamp = datetime.strptime(name, "%Y-%m-%d_%H%M%S")
    if timestamp.strftime("%Y-%m-%d_%H%M%S") != name:
        raise ValueError("실험 이름은 YYYY-MM-DD_HHMMSS 형식이어야 합니다.")
    destination = directory / "report.md"
    # 동시 평가 완료/CLI 실행이 더 오래된 인원으로 보고서를 덮어쓰지 않도록 직렬화합니다.
    lock = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        frames = load_completed(directory)
        count = len(frames)
        evaluation = (
            user_section(frames)
            if count >= REQUIRED_EVALUATORS
            else f"## 사용자 평가\n\n<사용자 평가 필요.({count}명/{REQUIRED_EVALUATORS}명)>"
        )
        content = f"# {name} 실험 결과 보고서\n\n{static_section(directory)}\n\n{evaluation}\n"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory, prefix=".report-", delete=False
            ) as output:
                temporary = Path(output.name)
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    finally:
        os.close(lock)
    return ReportResult(destination, count)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="실험 보고서 생성 또는 복구 (완료 평가자 3명 필요)"
    )
    parser.add_argument("--experiment", required=True, help="실험 이름: YYYY-MM-DD_HHMMSS")
    args = parser.parse_args()
    try:
        # 상위 경로나 임의 폴더를 실험 이름으로 받지 않습니다.
        if Path(args.experiment).name != args.experiment:
            raise ValueError("--experiment에는 경로가 아닌 실험 이름을 지정하세요.")
        config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
        result = build_report(PROJECT_ROOT / config["experiment"]["output_dir"] / args.experiment)
    except Exception as error:
        parser.exit(1, f"보고서 생성 실패: {error}\n")
    print(f"보고서 저장: {result.path}")
    if result.remaining_count:
        print(
            f"사용자 평가 {result.completed_count}명/3명: "
            f"보고서 생성을 위해 {result.remaining_count}명의 평가가 더 필요합니다."
        )
    else:
        print(f"사용자 평가 {result.completed_count}명: 보고서 생성이 완료되었습니다.")


if __name__ == "__main__":
    main()
