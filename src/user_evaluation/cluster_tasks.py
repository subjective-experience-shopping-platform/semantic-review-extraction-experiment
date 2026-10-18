"""우선 검토 군집 50%와 나머지 후보의 무작위 군집으로 문항을 구성합니다."""

import json

import numpy as np
import pandas as pd


def build_cluster_tasks(diagnostics: dict, count: int, seed: int) -> pd.DataFrame:
    if count < 1:
        raise ValueError("군집 평가 문항 수는 1 이상이어야 합니다.")
    clusters = pd.concat(
        [diagnostics[f"{name}_clusters"] for name in ("attribute", "aspect", "status")],
        ignore_index=True,
    ).sort_values(["stage", "cluster_id"], kind="stable")
    candidates = clusters.loc[
        clusters["member_count"].gt(1) & clusters["canonical_label"].notna()
    ].copy()
    if len(candidates) < count:
        raise ValueError(
            f"군집 평가 {count}문항을 중복 없이 만들기 위한 후보가 부족합니다: "
            f"대표명이 있고 구성원이 둘 이상인 후보 {len(candidates)}개"
        )
    candidates["risk_score"] = (
        candidates["cluster_max_distance"]
        + (1.0 - candidates["cluster_silhouette_mean"].fillna(0.0)) / 2.0
    )
    # 기존과 같이 각 단계의 위험 순위를 번갈아 사용합니다.
    ordered = candidates.sort_values(
        ["stage", "risk_score", "cluster_id"], ascending=[True, False, True], kind="stable"
    )
    ordered["stage_rank"] = ordered.groupby("stage", sort=False).cumcount()
    risky = ordered.sort_values(
        ["stage_rank", "risk_score", "stage", "cluster_id"],
        ascending=[True, False, True, True],
        kind="stable",
    ).head(count // 2)
    # 홀수 문항의 남은 한 개는 무작위 몫으로 배정합니다.
    random_pool = candidates.loc[~candidates["cluster_id"].isin(risky["cluster_id"])]
    remaining = count - len(risky)
    generator = np.random.default_rng(seed)
    positions = generator.choice(len(random_pool), size=remaining, replace=False)
    selected = pd.concat(
        [
            risky.assign(selection_reason="risky"),
            random_pool.iloc[positions].assign(selection_reason="random"),
        ],
        ignore_index=True,
    ).iloc[generator.permutation(count)]
    parents = {
        row.cluster_id: row.canonical_label if pd.notna(row.canonical_label) else row.medoid_label
        for row in diagnostics["aspect_clusters"].itertuples(index=False)
    }
    return pd.DataFrame(
        [
            {
                "cluster_validation_id": f"{row.stage}:{row.cluster_id}",
                "selection_reason": row.selection_reason,
                "stage": row.stage,
                "cluster_id": row.cluster_id,
                "parent_aspect_label": parents.get(row.aspect_cluster_id),
                "canonical_label": row.canonical_label,
                "members": json.dumps(list(row.member_expressions), ensure_ascii=False),
                "risk_score": float(row.risk_score),
            }
            for row in selected.itertuples(index=False)
        ]
    )
