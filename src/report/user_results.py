"""완료 응답을 검증하고 선호도, 리뷰 평점, 군집 평점을 집계합니다."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.user_evaluation.responses import (
    CLUSTER_SCORES,
    EXPERIMENTS,
    REVIEW_SCORES,
    validate_responses,
)

STAGE_LABELS = {
    "experiment_b": "B-속성",
    "experiment_d_aspect": "D-Aspect",
    "experiment_d_status": "D-Status",
}
EXPERIMENT_LABELS = {
    "A": "비구조화 출력 + 비정규화",
    "B": "비구조화 출력 + 정규화",
    "C": "구조화 출력 + 비정규화",
    "D": "구조화 출력 + 정규화",
}


def load_completed(directory: Path) -> list[pd.DataFrame]:
    evaluation = directory / "user_evaluation"
    review_tasks = pd.read_parquet(evaluation / "review_tasks.parquet").to_dict("records")
    cluster_tasks = pd.read_parquet(evaluation / "cluster_tasks.parquet").to_dict("records")
    digest = hashlib.sha256()
    for path in (
        directory / "data/dataset.parquet",
        evaluation / "review_tasks.parquet",
        evaluation / "cluster_tasks.parquet",
    ):
        digest.update(path.read_bytes())
    frames, reviewers = [], set()
    for path in sorted((evaluation / "completed_evaluators").glob("*.parquet")):
        if path.name.startswith(".") or not path.is_file():
            continue
        frame = pd.read_parquet(path)
        try:
            if frame.empty or not frame.evaluator_uuid.eq(path.stem).all():
                raise ValueError("평가자 ID와 파일 이름이 다릅니다.")
            if path.stem in reviewers or not frame.task_signature.eq(digest.hexdigest()).all():
                raise ValueError("중복 평가자이거나 실험 문항과 응답의 서명이 다릅니다.")
            if not frame.evaluation_level.isin(("review", "cluster")).all():
                raise ValueError("알 수 없는 평가 수준입니다.")
            review = frame.loc[frame.evaluation_level.eq("review")]
            cluster = frame.loc[frame.evaluation_level.eq("cluster")]
            normalized = validate_responses(
                review.to_dict("records"),
                cluster.to_dict("records"),
                review_tasks,
                cluster_tasks,
            )
            normalized["evaluator_uuid"] = path.stem
        except (KeyError, AttributeError, ValueError) as error:
            raise ValueError(f"완료 응답이 유효하지 않습니다: {path.name}: {error}") from error
        frames.append(normalized)
        reviewers.add(path.stem)
    return frames


def preference_intervals(review: pd.DataFrame) -> np.ndarray:
    """동일 리뷰의 모든 평가자의 단일 선택을 함께 10,000회 재표집합니다."""
    votes = review.groupby(["review_idx", "experiment"]).preferred.sum().unstack()
    votes = votes.reindex(columns=EXPERIMENTS).to_numpy(dtype=float)
    rng = np.random.default_rng(17171771)
    # 문항 수에 비례하는 거대한 배열을 만들지 않도록 배치별로 계산합니다.
    rates = []
    for _ in range(100):
        indices = rng.integers(0, len(votes), size=(100, len(votes)))
        sums = votes[indices].sum(axis=1)
        rates.append(sums / sums.sum(axis=1, keepdims=True) * 100)
    return np.quantile(np.concatenate(rates), [0.025, 0.975], axis=0)


def user_section(frames: list[pd.DataFrame]) -> str:
    combined = pd.concat(frames, ignore_index=True)
    review = combined.loc[combined.evaluation_level.eq("review")]
    cluster = combined.loc[combined.evaluation_level.eq("cluster")]
    counts = review.groupby("experiment").preferred.sum().reindex(EXPERIMENTS).astype(int)
    total = int(counts.sum())
    rates = counts / total * 100
    most = counts.index[counts.eq(counts.max())].tolist()
    least = counts.index[counts.eq(counts.min())].tolist()
    winner = ", ".join(f"실험 {name}({EXPERIMENT_LABELS[name]})" for name in most)
    blocks = review[["evaluator_uuid", "review_idx"]].drop_duplicates().shape[0]
    if total != blocks:
        raise ValueError(
            "선호 선택 수와 리뷰 평가 건수가 다릅니다. 리뷰마다 하나를 선택해야 합니다."
        )
    intervals = preference_intervals(review)
    cluster_tasks = cluster.drop_duplicates("cluster_validation_id")
    reasons = cluster_tasks.selection_reason.value_counts()
    lines = [
        "## 사용자 평가",
        "",
        f"{len(frames)}명의 평가자가 수행한 **리뷰 수준 평가**에서는 {winner}의 선호도가 "
        + ("공동으로 가장 높았습니다." if len(most) > 1 else "가장 높았습니다."),
        "",
        f"- 선택 수의 총합은 {total:,}입니다.({review.review_idx.nunique():,}개 리뷰 × "
        f"{len(frames)}명, 리뷰마다 하나 선택)",
        "- 평가자는 각 결과가 어느 실험에서 생성되었는지 알지 못하도록 "
        "평가자·리뷰별로 결과의 표시 순서를 무작위로 혼합합니다.",
        f"- 군집 수준 평가 문항은 실루엣 점수와 군집 최대 거리를 기반으로 한 "
        f"우선 검토 후보 {reasons.get('risky', 0):,}개와 나머지 후보의 "
        f"무작위 문항 {reasons.get('random', 0):,}개로 구성되었습니다.",
        "- 선택률은 전체 리뷰 평가 건수(리뷰 수 × 평가자 수) 대비 비율입니다. 95% bootstrap 구간은 "
        "동일 리뷰에 대한 모든 평가자의 선택을 묶어 리뷰를 10,000회 재표집한 "
        "percentile 구간입니다.(고정 seed: 17171771)",
        "",
        "| 실험 | 선택 수 | 선택률 | 95% bootstrap 구간 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for i, name in enumerate(EXPERIMENTS):
        lines.append(
            f"| {name} | {counts[name]:,} | {rates[name]:.2f}% | "
            f"[{intervals[0, i]:.2f}%, {intervals[1, i]:.2f}%] |"
        )
    lines += [
        "",
        f"또한 {', '.join('실험 ' + name for name in most)}는 전체 선택의 "
        f"{rates[most[0]]:.2f}%를 차지하며 가장 높은 선호도를 보였으며 "
        f"{', '.join('실험 ' + name for name in least)}는 전체 선택의 "
        f"{rates[least[0]]:.2f}%를 차지하며 가장 낮은 선호도를 보였습니다.",
        "",
        "| 평가 기준 | 실험 A | 실험 B | 실험 C | 실험 D |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    means = review.groupby("experiment")[list(REVIEW_SCORES)].mean().reindex(EXPERIMENTS)
    for key, (label, _) in REVIEW_SCORES.items():
        lines.append(f"| {label} | " + " | ".join(f"{value:.4f}" for value in means[key]) + " |")
    lines.append(
        "| 세 기준 단순평균 | " + " | ".join(f"{value:.4f}" for value in means.mean(axis=1)) + " |"
    )
    rating_count = len(cluster) * len(CLUSTER_SCORES)
    group_means = cluster.groupby("stage")[list(CLUSTER_SCORES)].mean()
    minimum = group_means.min().min()
    low = int(cluster[list(CLUSTER_SCORES)].isin((1, 2)).sum().sum())
    lines += [
        "",
        f"**군집 수준 평가**는 {len(frames)}명의 사용자로부터 총 {len(cluster):,}건의 "
        f"군집 평가, {rating_count:,}개의 평점을 수집하였습니다. 평가된 군집 유형의 "
        f"응집성과 대표명 적합성 평균은 1~5점 척도에서 모두 {minimum:.4f}점 이상이었습니다.",
        "",
        f"- 1점과 2점은 모든 평가자와 모든 기준에서 총 {low:,}개입니다. "
        "3점 / 4점 / 5점 열에는 이를 제외하며, 평점 수·평균·표본 SD·중앙값에는 포함합니다.",
        "- 군집 후보는 대표명이 있고 구성원이 둘 이상인 군집입니다. "
        "우선 검토 후보가 포함된 표본의 결과이므로 전체 군집의 품질로 일반화하지 않습니다.",
        "",
        "| 군집 유형 | 기준 | 평점 수 | 평균 | 표본 SD | 중앙값 | 3점 / 4점 / 5점 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage, label in STAGE_LABELS.items():
        group = cluster.loc[cluster.stage.eq(stage)]
        for key, (criterion, _) in CLUSTER_SCORES.items():
            values = group[key].astype(float)
            sd = f"{values.std(ddof=1):.4f}" if len(values) > 1 else "계산 불가"
            if values.empty:
                lines.append(
                    f"| {label} | {criterion.removeprefix('군집 ')} | 0 | — | — | — | 0 / 0 / 0 |"
                )
                continue
            distribution = " / ".join(str(int(values.eq(score).sum())) for score in (3, 4, 5))
            lines.append(
                f"| {label} | {criterion.removeprefix('군집 ')} | {len(values):,} | "
                f"{values.mean():.4f} | {sd} | {values.median():g} | {distribution} |"
            )
    return "\n".join(lines)
