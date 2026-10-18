"""필요하면 리뷰 데이터셋을 빌드한 뒤 전체 리뷰의 units를 추출합니다."""

import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    settings = config["origin_dataset"]
    archive_paths = [PROJECT_ROOT / path for path in settings["source_archives"]]
    missing_paths = [str(path) for path in archive_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("원본 ZIP 파일이 없습니다:\n" + "\n".join(missing_paths))

    dataset_path = PROJECT_ROOT / settings["output_path"]
    if dataset_path.exists():
        print(f"데이터셋이 이미 있어 빌드를 건너뛰고 속성을 추출합니다: {dataset_path}", flush=True)
    else:
        print(f"데이터셋이 없어 빌드를 수행합니다: {dataset_path}", flush=True)
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / "build_dataset.py")],
            cwd=PROJECT_ROOT,
            check=True,
        )

    print("속성 추출을 시작합니다. 완료되면 기존 두 units Parquet 파일을 덮어씁니다.", flush=True)
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "extract_attribute" / "extract_attribute.py")],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
