"""설정된 데이터셋 전체에서 Attribute Units와 Opinion Units를 추출합니다."""

import json
import logging
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml
from codex_cli import REASONING_EFFORT, extract_one
from contracts import FIELDS, output_schema
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def load_reviews(input_path: Path) -> list[dict]:
    dataset = pd.read_parquet(input_path)
    columns = ["idx", "review", "productName", "category"]
    if dataset.empty or not set(columns).issubset(dataset.columns):
        raise ValueError("입력 데이터셋은 비어 있지 않아야 하며 필수 4개 열이 필요합니다.")
    if dataset[columns].isna().any().any() or not dataset["idx"].is_unique:
        raise ValueError("입력에 빈 값이나 중복 리뷰 ID가 있습니다.")
    if not pd.api.types.is_integer_dtype(dataset["idx"]):
        raise ValueError("리뷰 ID는 정수여야 합니다.")
    for column in columns[1:]:
        for value in dataset[column]:
            if not isinstance(value, str) or not value or value != " ".join(value.split()):
                raise ValueError(f"{column}: 빈 텍스트나 정리되지 않은 공백이 있습니다.")
    return dataset[columns].to_dict(orient="records")


def extract_task(
    task: str, reviews: list[dict], temporary_dir: Path, model: str, cli_path: str, max_workers: int
) -> pd.DataFrame:
    template = (PROJECT_ROOT / "prompts" / f"{task}.md").read_text(encoding="utf-8")
    schema_path = temporary_dir / f"{task}_schema.json"
    schema_path.write_text(json.dumps(output_schema(task), ensure_ascii=False), encoding="utf-8")
    results_dir = temporary_dir / task
    results_dir.mkdir()
    results = [None] * len(reviews)
    logger.info("%s 시작: 리뷰 %s개, 최대 %s개 병렬 요청", task, len(reviews), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                extract_one, review, task, template, schema_path, results_dir, model, cli_path
            ): position
            for position, review in enumerate(reviews)
        }
        for future in tqdm(
            as_completed(futures), total=len(reviews), desc=task, unit="review", dynamic_ncols=True
        ):
            results[futures[future]] = future.result()
    # 병렬 요청이 끝난 순서가 아니라 입력 리뷰 순서로 결과를 저장합니다.
    records = [unit for batch in results for unit in batch]
    columns = ["review_idx", *FIELDS[task]]
    frame = pd.DataFrame(records, columns=columns)
    frame.insert(0, "idx", range(1, len(frame) + 1))
    frame = frame.astype({"idx": "int64", "review_idx": "int64"})
    for column in FIELDS[task]:
        frame[column] = frame[column].astype("string")
    logger.info("%s 완료: 리뷰 %s개, 추출 단위 %s개", task, len(reviews), len(frame))
    return frame


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    input_path = PROJECT_ROOT / config["origin_dataset"]["output_path"]
    settings = config["extract_attribute"]
    output_paths = {task: (PROJECT_ROOT / settings[task]).resolve() for task in FIELDS}
    if (
        len(set(output_paths.values())) != len(FIELDS)
        or input_path.resolve() in output_paths.values()
    ):
        raise ValueError("두 units 출력 경로는 서로 달라야 하며 입력 데이터셋과 겹칠 수 없습니다.")
    model = settings["model"]
    cli_path = settings["cli_path"]
    max_workers = settings.get("max_workers", 4)
    reviews = load_reviews(input_path)
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for destination in output_paths.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".extract_attribute-", dir=data_dir))
    logger.info(
        "입력: %s | 모델: %s (%s) | 임시 결과: %s",
        input_path,
        model,
        REASONING_EFFORT,
        temporary_dir,
    )
    try:
        # 두 방식의 전체 추출과 Parquet 생성이 끝난 뒤 최종 파일을 교체합니다.
        for task in FIELDS:
            frame = extract_task(task, reviews, temporary_dir, model, cli_path, max_workers)
            frame.to_parquet(temporary_dir / f"{task}.parquet", index=False, compression="zstd")
        for task in FIELDS:
            destination = output_paths[task]
            (temporary_dir / f"{task}.parquet").replace(destination)
            logger.info("저장: %s", destination)
    except Exception:
        logger.exception("추출 실패: 확인을 위해 임시 결과를 남깁니다: %s", temporary_dir)
        raise
    shutil.rmtree(temporary_dir)
    logger.info("두 추출 완료. 임시 JSON 폴더를 제거했습니다.")


if __name__ == "__main__":
    main()
