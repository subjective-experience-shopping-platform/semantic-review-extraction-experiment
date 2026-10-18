"""라벨링 ZIP에서 지정한 모니터 상품의 리뷰 데이터셋을 만듭니다."""

import json
import re
import zipfile
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRODUCT_NAMES = [
    "AOC 알파스캔 Q27G2S 게이밍 IPS 155 QHD 프리싱크 무결점",
    "삼성전자 F27T350",
    "BenQ GW2780 아이케어 무결점",
    "오디세이 삼성전자 G5 C32G54T",
    "LG전자 24MK430H",
    "오디세이 삼성전자 G5 C34G55T",
    "LG전자 32QN650",
    "삼성전자 U32J590",
    "오디세이 삼성전자 G5 C27G54T",
]

REQUIRED_FIELDS = ("Index", "RawText", "Source", "Domain", "MainCategory", "ProductName")


def clean_text(text: str) -> str:
    """탭, 줄바꿈, 연속 공백을 공백 하나로 바꿉니다."""
    return " ".join(text.split())


def is_korean_review(text: str) -> bool:
    """공백으로 나눈 단어 중 한글이 포함된 단어가 80% 이상인지 확인합니다."""
    words = text.split()
    if not words:
        return False
    korean_count = sum(bool(re.search(r"[가-힣]", word)) for word in words)
    return korean_count / len(words) >= 0.8


def load_reviews(archive_paths: list[Path]) -> pd.DataFrame:
    """기존 리뷰 필터를 통과한 지정 상품의 리뷰를 개수 제한 없이 읽습니다."""
    reviews = []
    for archive_path in archive_paths:
        with zipfile.ZipFile(archive_path, metadata_encoding="cp949") as archive:
            for name in sorted(archive.namelist()):
                if not name.endswith(".json"):
                    continue
                data = json.loads(archive.read(name).decode("utf-8-sig"))
                rows = data if isinstance(data, list) else [data]
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    if any(not str(row.get(field) or "").strip() for field in REQUIRED_FIELDS):
                        continue
                    if (
                        row["ProductName"] not in PRODUCT_NAMES
                        or row["Source"] != "쇼핑몰"
                        or row["MainCategory"] != "컴퓨터/주변기기"
                        or not is_korean_review(row["RawText"])
                    ):
                        continue
                    reviews.append(
                        {
                            "idx": int(row["Index"]),
                            "review": clean_text(row["RawText"]),
                            "productName": clean_text(row["ProductName"]),
                            "category": "모니터",
                        }
                    )
    return pd.DataFrame(reviews, columns=["idx", "review", "productName", "category"])


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    settings = config["origin_dataset"]
    archive_paths = [PROJECT_ROOT / path for path in settings["source_archives"]]
    output_path = PROJECT_ROOT / settings["output_path"]
    dataset = load_reviews(archive_paths)
    if dataset.empty:
        raise ValueError("추출 조건에 맞는 리뷰가 없습니다.")
    if not dataset["idx"].is_unique:
        raise ValueError("리뷰 식별자 idx가 중복되어 있습니다.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False, compression="zstd")
    print(f"저장 경로: {output_path}")
    print(f"상품 {dataset['productName'].nunique()}개, 리뷰 {len(dataset):,}개")
    print(dataset.groupby("productName", sort=False).size().to_string())
    print("\n무작위 5행:")
    print(
        dataset.sample(n=min(5, len(dataset))).to_json(
            orient="records", force_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
