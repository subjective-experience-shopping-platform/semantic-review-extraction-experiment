"""A–D 공통 리뷰를 무작위로 고르고, 군집 진단 문항과 함께 반환합니다."""

import json

import numpy as np
import pandas as pd

from .cluster_tasks import build_cluster_tasks


def sample_tasks(pool: pd.DataFrame, count: int, seed: int, name: str) -> pd.DataFrame:
    if count < 1 or len(pool) < count:
        raise ValueError(f"{name} 문항 {count}개가 필요하지만 후보는 {len(pool)}개입니다.")
    positions = np.random.default_rng(seed).choice(len(pool), size=count, replace=False)
    return pool.iloc[positions].reset_index(drop=True)


def build_review_tasks(results: dict, settings: dict) -> pd.DataFrame:
    display = {}
    for name, frame in results.items():
        if name in ("experiment_a", "experiment_b"):
            labels = frame["attribute"]
        else:
            labels = frame["aspect"] + frame["status"].map(
                lambda status: "" if pd.isna(status) else f" > {status}"
            )
        display[name] = frame.assign(label=labels)
    common = sorted(set.intersection(*(set(frame["review_idx"]) for frame in display.values())))
    review_tasks = pd.DataFrame({"review_idx": common})
    for experiment, frame in display.items():
        labels = (
            frame.sort_values("idx", kind="stable")
            .groupby("review_idx")["label"]
            # pandas의 Arrow 배열 변환 전에 Python 목록을 JSON으로 직렬화합니다.
            .agg(lambda values: json.dumps(list(dict.fromkeys(values)), ensure_ascii=False))
        )
        column = "result_" + experiment[-1]
        review_tasks[column] = review_tasks["review_idx"].map(labels)

    return sample_tasks(review_tasks, settings["task_count"], settings["random_seed"], "리뷰")


def build_tasks(results: dict, diagnostics: dict, settings: dict) -> dict[str, pd.DataFrame]:
    return {
        "review_tasks": build_review_tasks(results, settings),
        "cluster_tasks": build_cluster_tasks(
            diagnostics, settings["task_count"], settings["random_seed"]
        ),
    }
