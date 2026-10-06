from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from query_difficulty.artifacts import ArtifactStore
from query_difficulty.console import Console


TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ÿ0-9_#@%']+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


def join_fields(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


@dataclass(slots=True)
class DatasetBundle:
    publications: pd.DataFrame
    queries: dict[str, pd.DataFrame]


class DatasetManager:
    def __init__(self, data_dir: Path, store: ArtifactStore, console: Console) -> None:
        self.data_dir = data_dir
        self.store = store
        self.console = console

    def load(self) -> DatasetBundle:
        publications = self._load_publications()
        queries = {}
        for language in ["en", "de", "fr"]:
            for split_name in ["train", "dev", "test"]:
                raw_name = self._query_filename(language, split_name)
                path = self.data_dir / raw_name
                if not path.exists():
                    continue
                queries[f"{language}_{split_name}"] = self._load_queries(path, language, split_name)
        return DatasetBundle(publications=publications, queries=queries)

    def _load_publications(self) -> pd.DataFrame:
        cache_path = self.store.path("corpus", "publications.parquet")
        if cache_path.exists():
            return self.store.read_parquet(cache_path)
        source_path = self.data_dir / "collection_data.json"
        self.console.log(f"Preparing corpus from {source_path.name}")
        with source_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        frame = pd.DataFrame(payload)
        frame = frame.rename(columns={"pubkey": "document_id"})
        frame["title"] = frame["title"].fillna("")
        frame["abstract"] = frame["abstract"].fillna("")
        frame["venue"] = frame["venue"].fillna("")
        frame["authors"] = frame["authors"].fillna("")
        frame["document_text"] = [
            join_fields(title, abstract) for title, abstract in zip(frame["title"], frame["abstract"], strict=True)
        ]
        frame["title_tokens"] = frame["title"].map(tokenize)
        frame["abstract_tokens"] = frame["abstract"].map(tokenize)
        frame["authors_tokens"] = frame["authors"].map(tokenize)
        frame["venue_tokens"] = frame["venue"].map(tokenize)
        frame["document_tokens"] = frame["document_text"].map(tokenize)
        frame["document_token_text"] = frame["document_tokens"].map(" ".join)
        self.store.write_parquet(cache_path, frame)
        return frame

    def _load_queries(self, path: Path, language: str, split_name: str) -> pd.DataFrame:
        cache_path = self.store.path("queries", f"{language}_{split_name}.parquet")
        if cache_path.exists():
            return self.store.read_parquet(cache_path)
        self.console.log(f"Preparing queries from {path.name}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        frame = pd.DataFrame(payload)
        frame = frame.rename(columns={"index": "query_id", "pubkey": "document_id"})
        frame["language"] = language
        frame["split_name"] = split_name
        frame["split_key"] = f"{language}_{split_name}"
        frame["query_tokens"] = frame["text"].fillna("").map(tokenize)
        frame["query_token_text"] = frame["query_tokens"].map(" ".join)
        self.store.write_parquet(cache_path, frame)
        return frame

    @staticmethod
    def _query_filename(language: str, split_name: str) -> str:
        if split_name == "test":
            return f"final_{language}_test.json"
        return f"{language}_{split_name}.json"
