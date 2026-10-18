"""저장된 속성으로 A–D 군집화와 사용자 평가 문항 생성만 수행합니다."""

import hashlib
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from pandas.testing import assert_frame_equal

from src.clustering.clustering import build_experiments
from src.clustering.embedding import SentenceEncoder
from src.extract_attribute.contracts import FIELDS, SENTIMENTS
from src.report.build_report import build_report
from src.user_evaluation.tasks import build_tasks

PROJECT_ROOT = Path(__file__).resolve().parent


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_units(path: Path, task: str, reviews: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    expected = ["idx", "review_idx", *FIELDS[task]]
    if list(frame.columns) != expected or frame.empty:
        raise ValueError(f"{path.name}: 추출 결과의 열이 다르거나 비어 있습니다.")
    if frame["idx"].isna().any() or not frame["idx"].is_unique:
        raise ValueError(f"{path.name}: 중복되거나 비어 있는 idx가 있습니다.")
    if not frame["review_idx"].isin(reviews["idx"]).all():
        raise ValueError(f"{path.name}: 원본 데이터셋에 없는 리뷰를 참조합니다.")
    if not frame["sentiment"].isin(SENTIMENTS).all():
        raise ValueError(f"{path.name}: 유효하지 않은 감성 값이 있습니다.")
    for column in FIELDS[task]:
        values = frame[column].dropna() if column == "raw_status" else frame[column]
        if not values.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{path.name}: {column}에 빈 값이 있습니다.")
    return frame


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    experiment_root = PROJECT_ROOT / config["experiment"]["output_dir"]
    experiment_name = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S")
    experiment_dir = experiment_root / experiment_name
    if experiment_dir.exists():
        raise FileExistsError(f"같은 시각의 실험 폴더가 이미 있습니다: {experiment_dir}")
    input_paths = [
        PROJECT_ROOT / config["origin_dataset"]["output_path"],
        PROJECT_ROOT / config["extract_attribute"]["attribute_units"],
        PROJECT_ROOT / config["extract_attribute"]["opinion_units"],
    ]
    before = {path: file_hash(path) for path in input_paths}
    reviews = pd.read_parquet(input_paths[0])
    if reviews.empty or reviews["idx"].isna().any() or not reviews["idx"].is_unique:
        raise ValueError("리뷰 데이터셋이 비어 있거나 리뷰 ID가 중복되어 있습니다.")
    attribute = read_units(input_paths[1], "attribute_units", reviews)
    opinion = read_units(input_paths[2], "opinion_units", reviews)
    print(
        f"입력: 리뷰 {len(reviews):,}개, Attribute {len(attribute):,}행, Opinion {len(opinion):,}행",
        flush=True,
    )
    encoder = SentenceEncoder(config["embedding"], PROJECT_ROOT)
    results, diagnostics = build_experiments(attribute, opinion, encoder, config["clustering"])
    tasks = build_tasks(results, diagnostics, config["user_evaluation"])
    counts = tasks["cluster_tasks"]["selection_reason"].value_counts()
    print(
        f"군집 평가 {len(tasks['cluster_tasks'])}문항: "
        f"우선 검토 {counts.get('risky', 0)}개, 무작위 {counts.get('random', 0)}개",
        flush=True,
    )

    # 정규화 후에도 원래 열, 감성, null status가 유지되는지 확인합니다.
    for name, result in results.items():
        source = attribute if name[-1] in "ab" else opinion
        expected = source.loc[source["idx"].isin(result["idx"])]
        assert_frame_equal(
            result[source.columns].sort_values("idx").reset_index(drop=True),
            expected.sort_values("idx").reset_index(drop=True),
        )
        if "status" in result and not result["status"].isna().equals(result["raw_status"].isna()):
            raise ValueError(f"{name}: null status가 유지되지 않았습니다.")
    if any(file_hash(path) != checksum for path, checksum in before.items()):
        raise RuntimeError("처리 중 추출 원본이 변경되어 결과 저장을 중단합니다.")

    outputs = {
        Path("clustering") / f"{name}.parquet": frame
        for name, frame in {**results, **diagnostics}.items()
    }
    outputs.update(
        {Path("user_evaluation") / f"{name}.parquet": frame for name, frame in tasks.items()}
    )
    if set((experiment_dir / path).resolve() for path in outputs) & set(
        path.resolve() for path in input_paths
    ):
        raise ValueError("출력 경로가 추출 원본 경로와 겹칩니다.")
    # 저장이 끝난 실험만 선택 화면에 보이도록 임시 폴더를 완성한 뒤 이름을 바꿉니다.
    experiment_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".clustering-", dir=experiment_root))
    try:
        for relative_path, frame in outputs.items():
            destination = staging / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(destination, index=False, compression="zstd")
        # 실험에 사용한 입력 세 파일을 별도 data 폴더에 그대로 보존합니다.
        snapshot_dir = staging / "data"
        snapshot_dir.mkdir()
        for source, name in zip(
            input_paths,
            ("dataset.parquet", "attribute_units.parquet", "opinion_units.parquet"),
            strict=True,
        ):
            destination = snapshot_dir / name
            shutil.copy2(source, destination)
            if file_hash(destination) != before[source]:
                raise RuntimeError(f"실험 입력 사본이 원본과 다릅니다: {name}")
        if experiment_dir.exists():
            raise FileExistsError(f"같은 시각의 실험 폴더가 이미 있습니다: {experiment_dir}")
        staging.rename(experiment_dir)
        print(f"실험 저장: {experiment_dir}", flush=True)
        # 완성된 실험을 먼저 보존하므로 보고서 실패 시 CLI로 같은 실험을 복구할 수 있습니다.
        try:
            report = build_report(experiment_dir)
        except Exception as error:
            raise RuntimeError(
                f"실험은 저장되었으나 보고서 생성에 실패했습니다. "
                f"python src/report/build_report.py --experiment {experiment_name} 로 "
                f"재시도하세요: {error}"
            ) from error
        print(f"보고서 저장: {report.path} (사용자 평가 필요: 0명/3명)", flush=True)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
