"""원문과 네 결과를 비교한 뒤 군집을 평가하는 Streamlit 화면입니다."""

import hashlib
import html
import json
import random
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from src.report.build_report import build_report

from .experiments import list_experiments
from .responses import CLUSTER_SCORES, EXPERIMENTS, REVIEW_SCORES, save_responses
from .ui import (
    configure_score_shortcuts,
    next_button,
    render_chips,
    render_label,
    render_progress,
    render_score,
    render_styles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def return_to_selection() -> None:
    """이전 평가의 세션 상태를 비우고 다른 실험을 선택하도록 합니다."""
    keys = {
        "selected_experiment",
        "signature",
        "reviewer",
        "phase",
        "position",
        "review_responses",
        "cluster_responses",
        "completed_path",
        "report_result",
        "report_error",
    }
    for key in list(st.session_state):
        if key in keys or key.startswith(
            ("score::", "preferred::", "synced::", "cluster::", "score_button_", "keyboard_")
        ):
            del st.session_state[key]


def render_selection(experiments: list[dict]) -> None:
    if not experiments:
        st.info("평가할 실험이 없습니다. 먼저 run_experiment.py로 실험을 수행해 주세요.")
        return
    by_name = {experiment["name"]: experiment for experiment in experiments}
    _, center, _ = st.columns([1, 2, 1])
    with center, st.container(border=True, key="evaluation_card_selection"):
        st.subheader("평가할 실험 선택")
        st.write("리뷰의 추출 결과를 비교한 뒤, 군집의 구성과 대표명을 평가합니다.")
        selected = st.selectbox(
            "실험 목록",
            list(by_name),
            key="experiment_choice",
            format_func=lambda name: (
                f"{by_name[name]['label']} · 완료 {by_name[name]['completed_count']}명"
            ),
        )
        st.caption("모든 문항을 완료하면 응답이 저장됩니다.")
        if st.button("평가 시작", type="primary", width="stretch"):
            st.session_state.selected_experiment = selected
            st.rerun()


def input_signature(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def render_completion(experiment_dir: Path) -> None:
    """응답 저장 이후 동기적으로 보고서를 완성하고 종료 가능한 상태를 알립니다."""
    st.success("평가가 저장되었습니다. 참여해 주셔서 감사합니다.")
    if not st.session_state.get("report_result") and not st.session_state.get("report_error"):
        with st.status(
            "보고서 확인 및 생성 중입니다. 완료 안내가 나올 때까지 종료하지 마세요.",
            expanded=True,
        ) as status:
            try:
                result = build_report(experiment_dir)
            except Exception as error:
                st.session_state.report_error = str(error)
                status.update(label="보고서 생성 실패 · 평가 응답은 저장되었습니다.", state="error")
            else:
                st.session_state.report_result = result
                status.update(label="보고서 갱신이 완료되었습니다.", state="complete")
    result = st.session_state.get("report_result")
    if result:
        if result.remaining_count:
            st.info(
                f"보고서 생성을 위해 {result.remaining_count}명의 평가가 더 필요합니다. "
                f"(현재 {result.completed_count}명/3명) 이제 종료해도 됩니다."
            )
        else:
            st.success("보고서 생성이 완료되었습니다. 이제 종료해도 됩니다.")
        st.caption(f"보고서: {result.path}")
    else:
        st.error(f"보고서 생성 실패: {st.session_state.get('report_error')}")
        st.info("평가 응답은 저장되어 있습니다. 아래 버튼 또는 명시적 명령으로 재시도하세요.")
        st.code(f"python src/report/build_report.py --experiment {experiment_dir.name}")
        if st.button("보고서 생성 재시도"):
            st.session_state.pop("report_error", None)
            st.rerun()
    st.button("실험 선택으로 돌아가기", on_click=return_to_selection)


@st.cache_data
def load_tasks(paths: tuple[str, ...], signature: str) -> tuple[list[dict], list[dict]]:
    """파일 내용이 바뀌면 캐시도 바꾸며, 화면에는 Python 값만 전달합니다."""
    reviews, tasks, clusters = [
        pd.read_parquet(path, engine="pyarrow", use_threads=False) for path in paths
    ]
    if not reviews["idx"].is_unique or not tasks["review_idx"].is_unique:
        raise ValueError("입력에 중복 리뷰 ID가 있습니다.")
    joined = tasks.merge(
        reviews[["idx", "review", "productName"]].rename(columns={"idx": "review_idx"}),
        on="review_idx",
        how="left",
        validate="one_to_one",
    )
    if joined["review"].isna().any():
        raise ValueError("평가 문항에 해당하는 원문이 없습니다.")
    if tasks.empty or clusters.empty or not clusters["cluster_validation_id"].is_unique:
        raise ValueError("평가 문항이 비어 있거나 군집 문항 ID가 중복되어 있습니다.")
    if "task_type" in clusters and clusters["task_type"].eq("pair").any():
        raise ValueError(
            "이전 군집 쌍 문항입니다. run_experiment.py로 새 정책의 실험을 생성해 주세요."
        )
    if clusters["canonical_label"].isna().any():
        raise ValueError("대표명이 없는 군집 문항입니다. 새 정책의 실험을 생성해 주세요.")
    return joined.to_dict(orient="records"), clusters.to_dict(orient="records")


def sync_first_score(key: str, group_keys: list[str], marker: str) -> None:
    """동일한 결과들의 첫 점수만 함께 반영하고 이후 수정은 독립적으로 둡니다."""
    if st.session_state.get(marker) or st.session_state[key] is None:
        return
    for target in group_keys:
        if st.session_state.get(target) is None:
            st.session_state[target] = st.session_state[key]
    st.session_state[marker] = True


def render_review(task: dict, count: int) -> None:
    review_id = task["review_idx"]
    position = st.session_state.position
    render_progress("1단계 · 리뷰 수준 평가", position, count, "리뷰")
    with st.container(border=True, key="evaluation_card_review"):
        st.caption(task["productName"])
        st.markdown(
            f'<div class="review-text">{html.escape(task["review"])}</div>',
            unsafe_allow_html=True,
        )
    labels = {
        experiment: json.loads(task[f"result_{experiment.lower()}"]) for experiment in EXPERIMENTS
    }
    groups = {}
    for experiment, values in labels.items():
        groups.setdefault(tuple(sorted(set(values))), []).append(experiment)
    for experiment in EXPERIMENTS:
        for criterion in REVIEW_SCORES:
            st.session_state.setdefault(f"score::{review_id}::{experiment}::{criterion}", None)
    st.caption(
        "1점은 낮음, 5점은 높음입니다. 동일한 결과는 최초 점수만 함께 반영되며 이후 개별 수정할 수 있습니다."
    )
    order = list(EXPERIMENTS)
    random.Random(f"review-card-order-v1::{st.session_state.reviewer}::{review_id}").shuffle(order)
    configure_score_shortcuts(
        f"review::{review_id}",
        [
            (f"score::{review_id}::{experiment}::{criterion}", f"결과 {number} · {title}")
            for number, experiment in enumerate(order, start=1)
            for criterion, (title, _) in REVIEW_SCORES.items()
        ],
    )
    ratings = {}
    for number, (column, experiment) in enumerate(zip(st.columns(4), order, strict=True), start=1):
        with column, st.container(border=True, key=f"evaluation_card_result_{experiment}"):
            st.markdown(f'<div class="result-title">결과 {number}</div>', unsafe_allow_html=True)
            render_chips(labels[experiment], aligned=True)
            st.divider()
            ratings[experiment] = {}
            matching = groups[tuple(sorted(set(labels[experiment])))]
            for criterion, (title, help_text) in REVIEW_SCORES.items():
                key = f"score::{review_id}::{experiment}::{criterion}"
                group_keys = [f"score::{review_id}::{member}::{criterion}" for member in matching]
                marker = f"synced::{review_id}::{','.join(matching)}::{criterion}"
                ratings[experiment][criterion] = render_score(
                    key,
                    title,
                    help_text,
                    on_change=sync_first_score,
                    args=(key, group_keys, marker),
                )
    preferred = st.radio(
        "최종 선호 · 결과 하나를 선택해 주세요.",
        order,
        index=None,
        format_func=lambda experiment: f"결과 {order.index(experiment) + 1}",
        horizontal=True,
        key=f"preferred::{review_id}",
    )
    ready = preferred is not None and all(
        value is not None for scores in ratings.values() for value in scores.values()
    )
    if not ready:
        st.caption("모든 점수와 선호 결과 하나를 선택해 주세요.")
    last = position + 1 == count
    if next_button("군집 평가로 이동" if last else "다음 리뷰", ready):
        st.session_state.review_responses.extend(
            [
                {
                    "review_idx": review_id,
                    "experiment": experiment,
                    **ratings[experiment],
                    "preferred": experiment == preferred,
                }
                for experiment in EXPERIMENTS
            ]
        )
        st.session_state.position = 0 if last else position + 1
        if last:
            st.session_state.phase = "cluster"
        st.rerun()


def render_cluster(
    task: dict,
    reviews: list[dict],
    clusters: list[dict],
    output_dir: Path,
    signature: str,
    paths: list[Path],
) -> None:
    position, count = st.session_state.position, len(clusters)
    render_progress("2단계 · 군집 수준 평가", position, count, "문항")
    task_id = task["cluster_validation_id"]
    configure_score_shortcuts(
        f"cluster::{task_id}",
        [
            (f"cluster::{task_id}::{task['cluster_id']}::{criterion}", title)
            for criterion, (title, _) in CLUSTER_SCORES.items()
        ],
    )
    ratings = {}
    with st.container(border=True, key="evaluation_card_cluster"):
        if pd.notna(task["parent_aspect_label"]):
            render_chips([f"상위 속성 · {task['parent_aspect_label']}"])
        render_label(task["canonical_label"])
        st.markdown("**구성원**")
        render_chips(json.loads(task["members"]))
        st.divider()
        for column, (criterion, (title, description)) in zip(
            st.columns(len(CLUSTER_SCORES)), CLUSTER_SCORES.items(), strict=True
        ):
            with column:
                ratings[criterion] = render_score(
                    f"cluster::{task_id}::{task['cluster_id']}::{criterion}", title, description
                )
    response = {"cluster_validation_id": task_id, "cluster_id": task["cluster_id"], **ratings}
    last = position + 1 == count
    ready = all(value is not None for value in ratings.values())
    if not ready:
        st.caption("응답을 선택하면 다음 문항으로 이동할 수 있습니다.")
    if next_button("평가 완료 및 저장" if last else "다음 군집", ready):
        if last:
            try:
                if input_signature(paths) != signature:
                    raise ValueError(
                        "평가 중 입력 파일이 바뀌었습니다. 이 응답을 저장할 수 없습니다."
                    )
                destination = save_responses(
                    st.session_state.reviewer,
                    st.session_state.review_responses,
                    [*st.session_state.cluster_responses, response],
                    reviews,
                    clusters,
                    output_dir,
                    signature,
                )
            except Exception as error:
                st.error(f"응답 저장 실패: {error}")
                return
            st.session_state.completed_path = str(destination)
            st.session_state.phase = "completed"
        else:
            st.session_state.cluster_responses.append(response)
            st.session_state.position += 1
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="상품 속성 사용자 평가", layout="wide")
    render_styles()
    st.markdown(
        '<div class="evaluation-kicker">BLIND ATTRIBUTE EVALUATION</div>', unsafe_allow_html=True
    )
    st.title("상품 속성 사용자 평가")
    try:
        config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
        experiment_root = PROJECT_ROOT / config["experiment"]["output_dir"]
        experiments = list_experiments(experiment_root)
        if not st.session_state.get("selected_experiment"):
            render_selection(experiments)
            return
        selected = st.session_state.selected_experiment
        if selected not in {experiment["name"] for experiment in experiments}:
            raise FileNotFoundError("선택한 실험 결과를 찾을 수 없습니다.")
        experiment_dir = experiment_root / selected
        evaluation_dir = experiment_dir / "user_evaluation"
        paths = [
            experiment_dir / "data" / "dataset.parquet",
            evaluation_dir / "review_tasks.parquet",
            evaluation_dir / "cluster_tasks.parquet",
        ]
        signature = input_signature(paths)
        reviews, clusters = load_tasks(tuple(str(path) for path in paths), signature)
    except Exception as error:
        st.error(f"평가 문항을 읽지 못했습니다: {error}")
        st.info("먼저 run_experiment.py로 군집화와 평가 문항 생성을 완료해 주세요.")
        return
    st.caption(f"실험: {selected.replace('_', ' ')}")
    if "signature" not in st.session_state:
        st.session_state.update(
            signature=signature,
            reviewer=str(uuid.uuid4()),
            phase="review",
            position=0,
            review_responses=[],
            cluster_responses=[],
            completed_path=None,
        )
    elif st.session_state.signature != signature:
        st.error(
            "평가 중 데이터셋 또는 문항이 바뀌었습니다. 새 평가를 시작하려면 페이지를 새로 열어 주세요."
        )
        return
    if st.session_state.phase == "completed":
        render_completion(experiment_dir)
    elif st.session_state.phase == "review":
        render_review(reviews[st.session_state.position], len(reviews))
    else:
        render_cluster(
            clusters[st.session_state.position],
            reviews,
            clusters,
            evaluation_dir / "completed_evaluators",
            signature,
            paths,
        )
