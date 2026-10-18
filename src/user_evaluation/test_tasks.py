"""Arrow 문자열 입력에서도 평가 문항의 JSON이 올바르게 생성되는지 확인합니다."""

import json

import pandas as pd

from src.user_evaluation.tasks import build_review_tasks


def test_arrow_labels_keep_order_duplicates_and_null_status(tmp_path):
    # 실제 입력처럼 Parquet을 거친 Arrow 문자열을 사용합니다.
    frame = pd.DataFrame(
        {
            "idx": [1, 2, 3, 4],
            "review_idx": [11, 11, 11, 22],
            "attribute": pd.Series(["화질", "밝기", "화질", "무게"], dtype="string[pyarrow]"),
            "aspect": pd.Series(["화질", "밝기", "화질", "무게"], dtype="string[pyarrow]"),
            "status": pd.Series([None, "높음", None, "무거움"], dtype="string[pyarrow]"),
        }
    )
    path = tmp_path / "input.parquet"
    frame.to_parquet(path, index=False)
    frame = pd.read_parquet(path)
    results = {f"experiment_{name}": frame.copy() for name in "abcd"}
    review = build_review_tasks(results, {"task_count": 2, "random_seed": 17171771}).set_index(
        "review_idx"
    )
    for name in "ab":
        assert json.loads(review.at[11, f"result_{name}"]) == ["화질", "밝기"]
    for name in "cd":
        assert json.loads(review.at[11, f"result_{name}"]) == ["화질", "밝기 > 높음"]
        assert json.loads(review.at[22, f"result_{name}"]) == ["무게 > 무거움"]
