"""로컬 KR-SBERT로 문자열을 임베딩하고 같은 문자열의 계산을 재사용합니다."""

import json
from pathlib import Path

import numpy as np


class SentenceEncoder:
    def __init__(self, settings: dict, project_root: Path):
        import torch
        from sentence_transformers import SentenceTransformer

        model_path = project_root / settings["local_model_path"]
        if not (model_path / "modules.json").is_file():
            raise FileNotFoundError(f"로컬 임베딩 모델이 없습니다: {model_path}")
        manifest_path = model_path / "snapshot_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key in ("model_id", "revision"):
                if manifest[key] != settings[key]:
                    raise ValueError(f"로컬 모델의 {key}가 config.yaml과 다릅니다.")

        device = settings["device"]
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
        self.batch_size = settings["batch_size"]
        self.cache = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = sorted(set(texts) - self.cache.keys())
        if missing:
            vectors = self.model.encode(
                missing,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            ).astype(np.float32)
            self.cache.update(zip(missing, vectors, strict=True))
        return np.stack([self.cache[text] for text in texts])
