"""기존 평가 UI의 색상, 카드와 점수 버튼을 간결하게 재구성합니다."""

import html

import streamlit as st


def render_styles() -> None:
    st.markdown(
        """<style>
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 70% 10%, #eef1e8, transparent 38%), #f6f1e7;
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { max-width: 1500px; padding-top: 2.6rem; padding-bottom: 3rem; }
        .evaluation-kicker { color: #3f6f55; font-size: .75rem; font-weight: 800;
                             letter-spacing: .16em; margin-bottom: -.5rem; }
        div[class*="st-key-evaluation_card"] {
            background: rgba(255,255,255,.94); border-radius: 1.2rem;
            box-shadow: 0 12px 32px #1d2d220d;
        }
        div.st-key-evaluation_card_review {
            padding-top: .5rem; padding-bottom: 1.5rem; gap: .4rem;
        }
        .attribute-chip { display: inline-block; margin: 0 .35rem .45rem 0;
            padding: .32rem .62rem; border-radius: 999px; background: #edf2ed;
            color: #3f6f55; font-size: .85rem; font-weight: 650; overflow-wrap: anywhere; }
        .attribute-list { min-height: 13rem; }
        .review-text { font-size: 1.2rem; line-height: 1.7; color: #17251c;
                       overflow-wrap: anywhere; white-space: pre-wrap; }
        .result-title { font-size: 1.15rem; font-weight: 750; margin: .2rem 0 .6rem; }
        .score-heading { font-size: .9rem; font-weight: 750; color: #17251c; }
        .score-heading-active { color: #3f6f55; text-decoration: underline;
                                text-underline-offset: .3rem; }
        .score-help { font-size: .78rem; color: #626c64; margin: .15rem 0 .5rem; }
        .canonical-label-card { background: #3f6f55; color: white; border-radius: 1rem;
                                padding: 1rem 1.3rem; margin: .6rem 0 1rem; }
        .canonical-label-caption { font-size: .75rem; opacity: .8; margin-bottom: .25rem; }
        .canonical-label-value { font-size: 1.6rem; font-weight: 750; overflow-wrap: anywhere; }
        div[class*="st-key-score_button_"] button {
            aspect-ratio: 1; min-width: 0; min-height: 0; max-width: 3rem;
            padding: 0; border-radius: 999px;
        }
        div[class*="st-key-score_button_"] button kbd { display: none; }
        div[class*="st-key-score_button_"] button[kind="primary"] {
            background: #3f6f55; border-color: #3f6f55; color: white;
        }
        [data-testid="stButton"] button { border-radius: 999px; }
        [data-testid="stButton"] button[kind="primary"] {
            background: #17251c; border-color: #17251c;
        }
        [data-testid="stButton"] button[kind="primary"]:disabled {
            background: #e1e4e1; border-color: #d3d8d3; color: #626c64;
            opacity: 1;
        }
        @media (max-width: 640px) { .attribute-list { min-height: 0; } }
        </style>""",
        unsafe_allow_html=True,
    )


def render_chips(values: list[str], aligned: bool = False) -> None:
    chips = "".join(
        f'<span class="attribute-chip">{html.escape(str(value))}</span>'
        for value in values or ["속성 없음"]
    )
    st.markdown(
        f'<div class="{"attribute-list" if aligned else ""}">{chips}</div>',
        unsafe_allow_html=True,
    )


def render_label(label: str | None) -> None:
    st.markdown(
        '<div class="canonical-label-card"><div class="canonical-label-caption">대표명</div>'
        f'<div class="canonical-label-value">{html.escape(str(label))}</div></div>',
        unsafe_allow_html=True,
    )


def select_value(key: str, value: int, on_change=None, args: tuple = ()) -> None:
    st.session_state[key] = value
    if on_change is not None:
        on_change(*args)
    keys = st.session_state.get("keyboard_score_keys", [])
    if key in keys:
        st.session_state.keyboard_score_index = (keys.index(key) + 1) % len(keys)


def configure_score_shortcuts(scope: str, fields: list[tuple[str, str]]) -> None:
    """현재 문항의 화면 순서를 등록하고, 새 문항은 첫 항목부터 시작합니다."""
    if st.session_state.get("keyboard_score_scope") != scope:
        st.session_state.keyboard_score_index = 0
        st.session_state.keyboard_score_scope = scope
    st.session_state.keyboard_score_keys = [key for key, _ in fields]
    index = st.session_state.keyboard_score_index
    st.caption(f"숫자 키 1~5 · 현재: {fields[index][1]} · 선택 후 다음 항목으로 순환합니다.")


def render_score(key: str, title: str, description: str, on_change=None, args: tuple = ()):
    st.session_state.setdefault(key, None)
    keys = st.session_state.get("keyboard_score_keys", [])
    active = bool(keys) and keys[st.session_state.keyboard_score_index] == key
    heading_class = "score-heading score-heading-active" if active else "score-heading"
    st.markdown(f'<div class="{heading_class}">{html.escape(title)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="score-help">{html.escape(description)}</div>', unsafe_allow_html=True)
    for column, score in zip(st.columns(5, gap="small"), range(1, 6), strict=True):
        with column:
            st.button(
                str(score),
                key=f"score_button_{key}_{score}",
                type="primary" if st.session_state[key] == score else "secondary",
                width="stretch",
                shortcut=str(score) if active else None,
                wrap=True,
                on_click=select_value,
                args=(key, score, on_change, args),
            )
    return st.session_state[key]


def render_progress(stage: str, position: int, count: int, item: str) -> None:
    with st.container(border=True, key="evaluation_card_progress"):
        st.markdown(f"**{stage}**")
        st.progress(position / count, text=f"{position} / {count}개 완료 · {item} {position + 1}")


def next_button(label: str, ready: bool) -> bool:
    _, column = st.columns([4, 1])
    with column:
        return st.button(label, disabled=not ready, type="primary", width="stretch")
