"""실험 사본과 군집 진단에서 정적 보고서의 수치를 계산합니다."""

from pathlib import Path

import numpy as np
import pandas as pd

STAGES = (
    ("attribute", "B attribute"),
    ("aspect", "D aspect"),
    ("status", "D status"),
)


def percent(value: float | None, places: int = 1) -> str:
    return "계산 불가" if value is None else f"{value:.{places}f}%"


def static_section(directory: Path) -> str:
    data = directory / "data"
    reviews = pd.read_parquet(data / "dataset.parquet")
    attribute = pd.read_parquet(data / "attribute_units.parquet")
    opinion = pd.read_parquet(data / "opinion_units.parquet")
    eligible = pd.read_parquet(directory / "clustering/experiment_c.parquet")
    extracted = set(attribute.review_idx) | set(opinion.review_idx)
    if not extracted <= set(reviews.idx):
        raise ValueError("추출 결과가 실험 사본에 없는 리뷰를 참조합니다.")
    missing = len(reviews) - len(extracted)
    lines = [
        "## 정적 실험 결과",
        "",
        f"- 전체 {len(reviews):,}개 리뷰 중 {len(extracted):,}개 리뷰에서 속성(Units)이 "
        f"추출되었습니다. 속성이 추출되지 않은 리뷰는 {missing:,}개입니다. "
        "배송과 같이 상품에 대한 내용이 아닌 리뷰는 속성이 추출되지 않을 수 있습니다.",
        f"- 비구조화 프롬프트(실험 A,B)를 통해 {attribute.review_idx.nunique():,}개 리뷰에서 "
        f"{len(attribute):,}개의 속성을, 구조화 프롬프트를 통해 "
        f"{opinion.review_idx.nunique():,}개 리뷰에서 {len(opinion):,}개의 속성을 추출하였습니다. "
        f"이 중 실험 C,D에 사용한 속성은 {len(eligible):,}개이며, "
        f"제외된 속성은 {len(opinion) - len(eligible):,}개입니다.",
        "",
        "실험 B와 D에서 군집화를 통해 **속성을 정규화한 결과**는 다음과 같습니다.",
        "",
    ]
    metrics = []
    for name, label in STAGES:
        nodes = pd.read_parquet(directory / f"clustering/{name}_nodes.parquet")
        clusters = pd.read_parquet(directory / f"clustering/{name}_clusters.parquet")
        if len(nodes) != clusters.member_count.sum():
            raise ValueError(f"{label}: 노드 수와 군집 구성원 수가 다릅니다.")
        singleton = clusters.loc[clusters.member_count.eq(1)]
        unit_count = int(clusters.source_row_count.sum())
        singleton_units = int(singleton.source_row_count.sum())
        valid = clusters.loc[np.isfinite(clusters.cluster_silhouette_mean.astype(float))]
        valid_nodes = int(valid.member_count.sum())
        metrics.append(
            {
                "label": label,
                "nodes": len(nodes),
                "clusters": len(clusters),
                "compression": 100 * (1 - len(clusters) / len(nodes)) if len(nodes) else None,
                "singleton": len(singleton),
                "singleton_rate": 100 * len(singleton) / len(clusters) if len(clusters) else None,
                "singleton_units": singleton_units,
                "unit_count": unit_count,
                "unit_rate": 100 * singleton_units / unit_count if unit_count else None,
                "merged_rate": 100 * (1 - singleton_units / unit_count) if unit_count else None,
                "silhouette": (
                    float(np.average(valid.cluster_silhouette_mean, weights=valid.member_count))
                    if valid_nodes
                    else None
                ),
                "valid_nodes": valid_nodes,
            }
        )
    labels = "실험 B의 attribute, 실험 D의 aspect, 실험 D의 status"
    lines += [
        f"- {labels}는 각각 "
        + ", ".join(percent(m["compression"], 2) for m in metrics)
        + "의 속성 압축률을 보였습니다.",
        "- 생성된 군집 중 각각 "
        + ", ".join(percent(m["singleton_rate"]) for m in metrics)
        + "가 singleton, 즉 다른 속성과 병합되지 않은 단일 속성 군집으로 남았습니다.",
        "- 전체 추출 속성들(Units) 중 singleton 군집에 속한 비율은 B attribute, D aspect, D status에서 "
        "각각 "
        + ", ".join(percent(m["unit_rate"]) for m in metrics)
        + "입니다. 즉 각각 "
        + ", ".join(percent(m["merged_rate"]) for m in metrics)
        + " 속성들은 두 종류 이상의 표현이 통합한 군집에 속합니다.",
        "- 평균 노드 실루엣 점수는 각각 "
        + ", ".join(
            "계산 불가" if m["silhouette"] is None else f"{m['silhouette']:.3f}" for m in metrics
        )
        + "입니다.",
        "",
        "| 단계 | 고유 속성 수 | 군집 수 | 속성 압축률 | Singleton 군집 수(비율) | "
        "Singleton 군집에 속한 Units 수(비율) | 평균 node silhouette |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for m in metrics:
        silhouette = "계산 불가" if m["silhouette"] is None else f"{m['silhouette']:.3f}"
        lines.append(
            f"| {m['label']} | {m['nodes']:,} | {m['clusters']:,} | {percent(m['compression'], 2)} | "
            f"{m['singleton']:,}({percent(m['singleton_rate'])}) | "
            f"{m['singleton_units']:,}({percent(m['unit_rate'])}) | {silhouette} |"
        )
    return "\n".join(lines)
