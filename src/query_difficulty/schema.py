from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Publication:
    pubkey: int
    title: str
    abstract: str
    venue: str
    authors: str


@dataclass(slots=True)
class QueryRecord:
    query_id: int
    language: str
    split_name: str
    text: str
    pubkey: int


@dataclass(slots=True)
class SemanticRetrieverSpec:
    name: str
    model_name: str
    query_prefix: str
    document_prefix: str
    batch_size: int
    trust_remote_code: bool
    normalize_embeddings: bool
    revision: str | None = None
    code_revision: str | None = None
