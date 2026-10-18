"""Streamlit 사용자 평가 화면을 실행합니다."""

import os
import sys
from pathlib import Path


def main() -> None:
    os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx(suppress_warning=True) is None:
        os.execv(
            sys.executable,
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())],
        )
    from src.user_evaluation.app import main as run_app

    run_app()


if __name__ == "__main__":
    main()
