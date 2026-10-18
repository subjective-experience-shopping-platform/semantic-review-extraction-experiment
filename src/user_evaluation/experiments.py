"""완료된 실험 목록과 각 실험의 완료 평가자 수를 읽습니다."""

from datetime import datetime
from pathlib import Path


def list_experiments(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    experiments = []
    for directory in sorted(root.iterdir(), reverse=True):
        if not directory.is_dir():
            continue
        try:
            timestamp = datetime.strptime(directory.name, "%Y-%m-%d_%H%M%S")
        except ValueError:
            continue
        if timestamp.strftime("%Y-%m-%d_%H%M%S") != directory.name:
            continue
        required = [directory / "clustering" / f"experiment_{name}.parquet" for name in "abcd"]
        required += [
            directory / "data" / name
            for name in (
                "dataset.parquet",
                "attribute_units.parquet",
                "opinion_units.parquet",
            )
        ]
        required += [
            directory / "user_evaluation" / f"{name}_tasks.parquet"
            for name in ("review", "cluster")
        ]
        if not all(path.is_file() for path in required):
            continue
        completed = directory / "user_evaluation" / "completed_evaluators"
        count = sum(
            1
            for path in completed.glob("*.parquet")
            if path.is_file() and not path.name.startswith(".")
        )
        experiments.append(
            {
                "name": directory.name,
                "label": timestamp.strftime("%Y-%m-%d %H%M%S"),
                "completed_count": count,
            }
        )
    return experiments
