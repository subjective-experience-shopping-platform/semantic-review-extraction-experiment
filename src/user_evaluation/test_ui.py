"""장식용 HTML에서 입력 텍스트가 실행되지 않도록 확인합니다."""

from streamlit.testing.v1 import AppTest


def test_chip_and_label_escape_input_html():
    app = AppTest.from_string(
        "from src.user_evaluation.ui import render_chips, render_label\n"
        "render_chips(['<script>alert(1)</script>', '화질 & 밝기'])\n"
        "render_label('<img src=x onerror=alert(1)>')\n"
    ).run()
    assert not app.exception
    markup = "\n".join(element.value for element in app.markdown)
    assert "<script>" not in markup
    assert "<img " not in markup
    assert "&lt;script&gt;" in markup
    assert "&lt;img " in markup
    assert "화질 &amp; 밝기" in markup


def test_new_question_resets_keyboard_order():
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from src.user_evaluation.ui import configure_score_shortcuts, render_score\n"
        "scope = st.selectbox('문항', ['review-1', 'review-2'])\n"
        "configure_score_shortcuts(scope, [(scope + '-a', '첫 지표'), (scope + '-b', '두 번째 지표')])\n"
        "render_score(scope + '-a', '첫 지표', '설명')\n"
        "render_score(scope + '-b', '두 번째 지표', '설명')\n"
    ).run()
    app.button(key="score_button_review-1-a_2").click().run()
    assert app.session_state["keyboard_score_index"] == 1
    app.selectbox[0].set_value("review-2").run()
    assert not app.exception
    assert app.session_state["keyboard_score_index"] == 0
    assert app.button(key="score_button_review-2-a_5").proto.shortcut == "5"
    assert app.session_state["review-2-a"] is None
