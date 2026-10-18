"""고유 표현을 complete linkage로 묶고 실제 구성원 중 대표명을 선택합니다."""

from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

from .embedding import SentenceEncoder
from .rules import (
    ASPECT_DISALLOWED_LABELS,
    STATUS_NEGATION_SUFFIXES,
    STATUS_NEGATION_TOKENS,
    STATUS_OPPOSITION_GROUPS,
    STATUS_OPPOSITION_SUFFIX_PAIRS,
)


def negation_core(text: str) -> tuple[bool, str]:
    for suffix in STATUS_NEGATION_SUFFIXES:
        if text.endswith(suffix) and text[: -len(suffix)].strip():
            return True, text[: -len(suffix)].strip()
    tokens = text.split()
    for position, token in enumerate(tokens):
        if token in STATUS_NEGATION_TOKENS and len(tokens) > 1:
            return True, " ".join(tokens[:position] + tokens[position + 1 :])
    return False, text


def opposite_status(left: str, right: str, settings: dict) -> bool:
    for side_a, side_b in STATUS_OPPOSITION_GROUPS:
        if (left in side_a and right in side_b) or (right in side_a and left in side_b):
            return True
    similarity = settings["status_core_similarity_threshold"]
    for a, b in STATUS_OPPOSITION_SUFFIX_PAIRS:
        for first, second in ((left, right), (right, left)):
            if first.endswith(a) and second.endswith(b):
                core_a, core_b = first[: -len(a)].strip(), second[: -len(b)].strip()
                if (
                    core_a
                    and core_b
                    and SequenceMatcher(None, core_a, core_b).ratio() >= similarity
                ):
                    return True
    left_negative, left_core = negation_core(left)
    right_negative, right_core = negation_core(right)
    if left_negative != right_negative:
        core, positive = (left_core, right) if left_negative else (right_core, left)
        return SequenceMatcher(None, core, positive).ratio() >= similarity
    return False


def select_label(
    nodes: pd.DataFrame,
    distances: np.ndarray,
    column: str,
    role: str,
    settings: dict,
    forbidden: set[str],
    naming_max_distance: float,
) -> tuple[int | None, str, int]:
    """중심 후보 중 중심성 70%·리뷰 빈도 30%로 실제 대표명을 선택합니다."""
    size = len(nodes)
    expressions = nodes[column].tolist()
    means = distances.sum(axis=1) / max(size - 1, 1)
    order = sorted(range(size), key=lambda i: (means[i], expressions[i]))
    medoid = order[0]
    if size == 1:
        return 0, "singleton_inherited", medoid
    if float(distances.max()) > naming_max_distance + 1e-12:
        return None, "cohesion_review_required", medoid
    candidate_count = 2 if size == 2 else 3 if size < 10 else 5
    candidates = [
        i
        for i in order[:candidate_count]
        if (role != "aspect" or expressions[i] not in ASPECT_DISALLOWED_LABELS)
        and (role != "status" or expressions[i] not in forbidden)
    ]
    if not candidates:
        return None, "no_role_eligible_candidate", medoid
    support = nodes["unique_review_count"].to_numpy()
    frequency = np.log1p(support[candidates])
    frequency /= frequency.max()
    alpha = settings["centrality_weight_alpha"]
    scores = {
        i: alpha * np.clip(1 - means[i] / 2, 0, 1) + (1 - alpha) * frequency[position]
        for position, i in enumerate(candidates)
    }
    chosen = min(candidates, key=lambda i: (-scores[i], means[i], -support[i], expressions[i]))
    return chosen, "selected", medoid


def cluster_expressions(
    frame: pd.DataFrame,
    column: str,
    encoder: SentenceEncoder,
    settings: dict,
    threshold: float,
    stage: str,
    prefix: str,
    role: str,
    boundary: str | None = None,
    forbidden_labels: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """표현마다 노드 하나를 만들고, status는 상위 aspect 내부에서만 군집화합니다."""
    keys = [column] if boundary is None else [boundary, column]
    nodes = (
        frame.loc[frame[column].notna()]
        .groupby(keys, sort=True, observed=True)
        .agg(
            source_row_count=("idx", "size"),
            review_ids=("review_idx", lambda ids: tuple(sorted(set(ids)))),
        )
        .reset_index()
    )
    nodes["unique_review_count"] = nodes["review_ids"].map(len)
    for name in ("cluster_id", "canonical_label", "naming_status"):
        nodes[name] = pd.Series(pd.NA, index=nodes.index, dtype="string")
    nodes["mapping_applied"] = False
    nodes["mapping_distance"] = 0.0
    groups = [(None, nodes)] if boundary is None else list(nodes.groupby(boundary, sort=True))
    clusters = []
    counter = 0
    for parent, group in tqdm(groups, desc=stage, unit="group", dynamic_ncols=True):
        if group.empty:
            continue
        group = group.reset_index()
        texts = group[column].tolist()
        vectors = encoder.encode(texts)
        distances = np.clip(cosine_distances(vectors), 0.0, 2.0)
        np.fill_diagonal(distances, 0.0)
        fit_distances = distances.copy()
        if role == "status":
            for left in range(len(texts)):
                for right in range(left + 1, len(texts)):
                    if opposite_status(texts[left], texts[right], settings):
                        fit_distances[left, right] = fit_distances[right, left] = 2.0
        labels = np.zeros(len(group), dtype=int)
        if len(group) > 1:
            labels = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage="complete",
                distance_threshold=threshold,
                compute_full_tree=True,
            ).fit_predict(fit_distances if role == "status" else distances)
        memberships = [np.flatnonzero(labels == label) for label in set(labels)]
        memberships.sort(key=lambda positions: min(texts[i] for i in positions))
        # status의 실루엣도 동일한 상위 aspect 안에서만 계산합니다.
        silhouettes = (
            silhouette_samples(distances, labels, metric="precomputed")
            if 1 < len(memberships) < len(group)
            else np.full(len(group), np.nan)
        )
        forbidden = (forbidden_labels or {}).get(parent, set())
        for positions in memberships:
            counter += 1
            cluster_id = f"{prefix}-{counter:06d}"
            members = group.iloc[positions].reset_index(drop=True)
            local_distances = distances[np.ix_(positions, positions)]
            maximum_distance = float(local_distances.max())
            if maximum_distance > threshold + 1e-6:
                raise ValueError(f"군집 최대 거리가 임계값을 초과합니다: {cluster_id}")
            chosen, naming_status, medoid = select_label(
                members, local_distances, column, role, settings, forbidden, threshold
            )
            canonical = members.iloc[chosen][column] if chosen is not None else None
            for local_position, row in enumerate(members.to_dict(orient="records")):
                index = row["index"]
                nodes.at[index, "cluster_id"] = cluster_id
                nodes.at[index, "canonical_label"] = canonical
                nodes.at[index, "naming_status"] = naming_status
                nodes.at[index, "mapping_applied"] = (
                    canonical is not None and canonical != row[column]
                )
                nodes.at[index, "mapping_distance"] = (
                    float(local_distances[local_position, chosen])
                    if canonical is not None and canonical != row[column]
                    else 0.0
                )
            clusters.append(
                {
                    "stage": stage,
                    "cluster_id": cluster_id,
                    "canonical_label": canonical,
                    "medoid_label": members.iloc[medoid][column],
                    "naming_status": naming_status,
                    "member_count": len(members),
                    "member_expressions": members[column].tolist(),
                    "source_row_count": int(members["source_row_count"].sum()),
                    "unique_review_count": len(set().union(*members["review_ids"])),
                    "cluster_max_distance": maximum_distance,
                    "cluster_silhouette_mean": (
                        float(silhouettes[positions].mean())
                        if np.isfinite(silhouettes[positions]).all()
                        else None
                    ),
                    "distance_threshold": threshold,
                    "aspect_cluster_id": parent,
                }
            )
    cluster_columns = [
        "stage",
        "cluster_id",
        "canonical_label",
        "medoid_label",
        "naming_status",
        "member_count",
        "member_expressions",
        "source_row_count",
        "unique_review_count",
        "cluster_max_distance",
        "cluster_silhouette_mean",
        "distance_threshold",
        "aspect_cluster_id",
    ]
    return nodes.drop(columns="review_ids"), pd.DataFrame(clusters, columns=cluster_columns)


def apply_labels(
    frame: pd.DataFrame,
    nodes: pd.DataFrame,
    keys: list[str],
    target: str,
) -> pd.DataFrame:
    """원래 열을 유지하면서 정규화 열과 군집 연결 정보를 추가합니다."""
    fields = [
        "cluster_id",
        "canonical_label",
        "naming_status",
        "mapping_applied",
        "mapping_distance",
    ]
    names = {field: f"{target}_{field}" for field in fields}
    result = frame.merge(
        nodes[keys + fields].rename(columns=names),
        on=keys,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    result[target] = result[f"{target}_canonical_label"].fillna(result[keys[-1]])
    return result.drop(columns=f"{target}_canonical_label")


def build_experiments(
    attribute: pd.DataFrame, opinion: pd.DataFrame, encoder: SentenceEncoder, settings: dict
) -> tuple[dict, dict]:
    eligible = opinion.loc[~opinion["raw_aspect"].isin(settings["excluded_raw_aspects"])].copy()
    if eligible.empty:
        raise ValueError("C/D에 사용할 Opinion Units가 없습니다.")
    print(
        f"C/D 입력 {len(eligible):,}행, 실험에서 제외 {len(opinion) - len(eligible):,}행 (추출 원본은 유지)",
        flush=True,
    )
    b_nodes, b_clusters = cluster_expressions(
        attribute,
        "raw_attribute",
        encoder,
        settings,
        settings["attribute_distance_threshold"],
        "experiment_b",
        "B",
        "aspect",
    )
    aspect_nodes, aspect_clusters = cluster_expressions(
        eligible,
        "raw_aspect",
        encoder,
        settings,
        settings["aspect_distance_threshold"],
        "experiment_d_aspect",
        "D-A",
        "aspect",
    )
    d = apply_labels(eligible, aspect_nodes, ["raw_aspect"], "aspect")
    forbidden = {
        cluster: set(group["raw_aspect"]) | set(group["canonical_label"].dropna())
        for cluster, group in aspect_nodes.groupby("cluster_id")
    }
    status_nodes, status_clusters = cluster_expressions(
        d,
        "raw_status",
        encoder,
        settings,
        settings["status_distance_threshold"],
        "experiment_d_status",
        "D-S",
        "status",
        boundary="aspect_cluster_id",
        forbidden_labels=forbidden,
    )
    results = {
        "experiment_a": attribute.assign(attribute=attribute["raw_attribute"]),
        "experiment_b": apply_labels(attribute, b_nodes, ["raw_attribute"], "attribute"),
        "experiment_c": eligible.assign(
            aspect=eligible["raw_aspect"], status=eligible["raw_status"]
        ),
        "experiment_d": apply_labels(
            d, status_nodes, ["aspect_cluster_id", "raw_status"], "status"
        ),
    }
    diagnostics = {
        "attribute_nodes": b_nodes,
        "attribute_clusters": b_clusters,
        "aspect_nodes": aspect_nodes,
        "aspect_clusters": aspect_clusters,
        "status_nodes": status_nodes,
        "status_clusters": status_clusters,
    }
    return results, diagnostics
