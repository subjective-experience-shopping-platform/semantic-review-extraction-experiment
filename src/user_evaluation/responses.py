"""리뷰와 군집 평가가 모두 완료된 한 평가자의 응답만 저장합니다."""

from pathlib import Path

import pandas as pd

EXPERIMENTS = ("A", "B", "C", "D")
REVIEW_SCORES = {
    "factual_faithfulness": ("사실 충실성", "리뷰에서 실제로 언급된 내용인가?"),
    "core_explanatory_power": ("핵심 설명력", "핵심 구매 판단 요소를 설명하는가?"),
    "information_coverage": ("정보 포괄성", "중요 정보를 충분히 포괄하는가?"),
}
CLUSTER_SCORES = {
    "cohesion": ("군집 응집성", "모든 구성원이 하나의 동일한 개념인가?"),
    "canonical_label_fit": ("대표명 적합성", "대표명이 구성원 전체를 잘 설명하는가?"),
}


def validate_responses(
    review_rows: list[dict],
    cluster_rows: list[dict],
    review_tasks: list[dict],
    cluster_tasks: list[dict],
) -> pd.DataFrame:
    """저장과 보고서가 같은 완료 조건을 사용하도록 응답을 검증합니다."""
    review = pd.DataFrame(review_rows)
    cluster = pd.DataFrame(cluster_rows)
    expected_reviews = {row["review_idx"] for row in review_tasks}
    expected_clusters = {task["cluster_validation_id"] for task in cluster_tasks}
    if review.empty or cluster.empty:
        raise ValueError("리뷰와 군집 평가를 모두 완료해야 합니다.")
    if set(review["review_idx"]) != expected_reviews or len(review) != 4 * len(expected_reviews):
        raise ValueError("전체 리뷰 평가가 완료되지 않았습니다.")
    if review.duplicated(["review_idx", "experiment"]).any():
        raise ValueError("리뷰 평가에 중복 응답이 있습니다.")
    if any(
        set(group["experiment"]) != set(EXPERIMENTS) for _, group in review.groupby("review_idx")
    ):
        raise ValueError("리뷰마다 A–D 네 결과를 모두 평가해야 합니다.")
    if set(cluster["cluster_validation_id"]) != expected_clusters or len(cluster) != len(
        expected_clusters
    ):
        raise ValueError("전체 군집 평가가 완료되지 않았거나 중복 응답이 있습니다.")
    if cluster["cluster_validation_id"].duplicated().any():
        raise ValueError("군집 평가에 중복 응답이 있습니다.")
    for column in REVIEW_SCORES:
        if not review[column].isin(range(1, 6)).all():
            raise ValueError(f"{column}: 평점은 1~5의 정수여야 합니다.")
    by_id = cluster.set_index("cluster_validation_id")
    cluster_records = []
    for task in cluster_tasks:
        if task.get("task_type") == "pair" or pd.isna(task["canonical_label"]):
            raise ValueError("한 군집과 대표명이 있는 새 정책의 문항으로 평가해야 합니다.")
        response = by_id.loc[task["cluster_validation_id"]]
        if response.get("cluster_id") != task["cluster_id"]:
            raise ValueError("문항과 응답의 군집 ID가 일치하지 않습니다.")
        for criterion in CLUSTER_SCORES:
            value = response.get(criterion)
            if pd.isna(value) or value not in range(1, 6):
                raise ValueError(f"{criterion}: 평점은 1~5의 정수여야 합니다.")
        cluster_records.append(
            {
                "cluster_validation_id": task["cluster_validation_id"],
                "stage": task["stage"],
                "cluster_id": task["cluster_id"],
                "selection_reason": task.get("selection_reason", "random"),
                **{key: response[key] for key in CLUSTER_SCORES},
            }
        )
    cluster = pd.DataFrame(cluster_records)
    if review["preferred"].isna().any() or not pd.api.types.is_bool_dtype(review["preferred"]):
        raise ValueError("선호 선택은 비어 있지 않은 boolean이어야 합니다.")
    if review.groupby("review_idx")["preferred"].sum().ne(1).any():
        raise ValueError("리뷰마다 선호 결과를 정확히 하나 선택해야 합니다.")

    completed = pd.concat(
        [review.assign(evaluation_level="review"), cluster.assign(evaluation_level="cluster")],
        ignore_index=True,
        sort=False,
    )
    completed = completed.astype(
        {
            "review_idx": "Int64",
            **{column: "Int64" for column in (*REVIEW_SCORES, *CLUSTER_SCORES)},
            "preferred": "boolean",
        }
    )
    return completed


def save_responses(
    reviewer: str,
    review_rows: list[dict],
    cluster_rows: list[dict],
    review_tasks: list[dict],
    cluster_tasks: list[dict],
    output_dir: Path,
    task_signature: str,
) -> Path:
    completed = validate_responses(review_rows, cluster_rows, review_tasks, cluster_tasks)
    completed.insert(0, "evaluator_uuid", reviewer)
    completed.insert(1, "task_signature", task_signature)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{reviewer}.parquet"
    if destination.exists():
        raise FileExistsError(f"이미 저장된 응답입니다: {destination}")
    temporary = output_dir / f".{reviewer}.parquet"
    try:
        completed.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
