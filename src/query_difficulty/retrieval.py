from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import joblib
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm.auto import tqdm

from query_difficulty.artifacts import ArtifactStore
from query_difficulty.config import ProjectConfig
from query_difficulty.console import Console
from query_difficulty.dataset import join_fields, tokenize
from query_difficulty.schema import SemanticRetrieverSpec


LEXICAL_RETRIEVERS = {"bm25", "tfidf", "keyword_overlap", "jaccard"}
RETRIEVER_SCOPE_SEPARATOR = "__"
DEFAULT_DOCUMENT_SCOPE = "title_abstract"
DOCUMENT_SCOPE_FIELDS = {
    "title_abstract": ("title", "abstract"),
    "title_abstract_venue": ("title", "abstract", "venue"),
    "all_fields": ("title", "abstract", "venue", "authors"),
    "title": ("title",),
    "abstract": ("abstract",),
    "venue": ("venue",),
    "authors": ("authors",),
}


def compose_retriever_name(base_name: str, document_scope: str) -> str:
    if document_scope == DEFAULT_DOCUMENT_SCOPE:
        return base_name
    return f"{base_name}{RETRIEVER_SCOPE_SEPARATOR}{document_scope}"


def split_retriever_name(retriever_name: str) -> tuple[str, str]:
    if RETRIEVER_SCOPE_SEPARATOR not in retriever_name:
        return retriever_name, DEFAULT_DOCUMENT_SCOPE
    base_name, document_scope = retriever_name.rsplit(RETRIEVER_SCOPE_SEPARATOR, 1)
    return base_name, document_scope


def base_retriever_name(retriever_name: str) -> str:
    return split_retriever_name(retriever_name)[0]


def retriever_document_scope(retriever_name: str) -> str:
    return split_retriever_name(retriever_name)[1]


def is_lexical_retriever(retriever_name: str) -> bool:
    return base_retriever_name(retriever_name) in LEXICAL_RETRIEVERS


def _scopes_for_retriever(
    base_name: str,
    document_scopes: list[str],
    full_scope_retrievers: list[str] | None,
) -> list[str]:
    if full_scope_retrievers and base_name not in full_scope_retrievers:
        return [DEFAULT_DOCUMENT_SCOPE] if DEFAULT_DOCUMENT_SCOPE in document_scopes else []
    return document_scopes


def expand_retriever_names(
    lexical_retrievers: list[str],
    semantic_retrievers: list[str],
    document_scopes: list[str],
    full_scope_retrievers: list[str] | None = None,
) -> list[str]:
    names: list[str] = []
    for base_name in lexical_retrievers + semantic_retrievers:
        for document_scope in _scopes_for_retriever(base_name, document_scopes, full_scope_retrievers):
            names.append(compose_retriever_name(base_name, document_scope))
    return names


def ordered_enabled_retrievers(
    lexical_retrievers: list[str],
    semantic_retrievers: list[str],
    document_scopes: list[str],
    full_scope_retrievers: list[str] | None = None,
) -> list[str]:
    names: list[str] = []
    remaining: list[str] = []
    prioritized: list[str] = []

    for base_name in lexical_retrievers + semantic_retrievers:
        for document_scope in _scopes_for_retriever(base_name, document_scopes, full_scope_retrievers):
            remaining.append(compose_retriever_name(base_name, document_scope))

    if DEFAULT_DOCUMENT_SCOPE in document_scopes:
        if "bm25" in lexical_retrievers:
            prioritized.append(compose_retriever_name("bm25", DEFAULT_DOCUMENT_SCOPE))
        for base_name in semantic_retrievers:
            prioritized.append(compose_retriever_name(base_name, DEFAULT_DOCUMENT_SCOPE))
        for base_name in lexical_retrievers:
            if base_name == "bm25":
                continue
            prioritized.append(compose_retriever_name(base_name, DEFAULT_DOCUMENT_SCOPE))

    seen: set[str] = set()
    for name in prioritized + remaining:
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _top_k(scores: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    if limit >= len(scores):
        order = np.argsort(scores)[::-1]
        return order, scores[order]
    candidate_index = np.argpartition(scores, -limit)[-limit:]
    candidate_scores = scores[candidate_index]
    order = np.argsort(candidate_scores)[::-1]
    return candidate_index[order], candidate_scores[order]


@dataclass(slots=True)
class LexicalArtifacts:
    name: str
    model: Any
    vectorizer: Any | None
    matrix: sparse.spmatrix | None
    count_vectorizer: Any | None
    count_matrix: sparse.spmatrix | None
    binary_matrix: sparse.spmatrix | None
    document_lengths: np.ndarray
    vocabulary: dict[str, int]


@dataclass(slots=True)
class SharedLexicalArtifacts:
    count_vectorizer: Any
    count_matrix: sparse.spmatrix
    binary_matrix: sparse.spmatrix
    document_lengths: np.ndarray
    vocabulary: dict[str, int]


class RetrievalManager:
    def __init__(
        self,
        config: ProjectConfig,
        store: ArtifactStore,
        console: Console,
        publications: pd.DataFrame,
        queries: dict[str, pd.DataFrame],
    ) -> None:
        self.config = config
        self.store = store
        self.console = console
        self.publications = publications.reset_index(drop=True)
        self.queries = queries
        self.document_ids = self.publications["document_id"].to_numpy(dtype=np.int64)
        self._ranking_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._lexical_artifact_cache: dict[str, LexicalArtifacts] = {}
        self._shared_lexical_cache: dict[str, SharedLexicalArtifacts] = {}
        self._tfidf_payload_cache: dict[str, dict[str, Any]] = {}
        self._bm25_model_cache: dict[str, Any] = {}
        self._semantic_embedding_cache: dict[str, tuple[SemanticRetrieverSpec, np.ndarray]] = {}
        self._query_embedding_cache: dict[tuple[str, str], np.ndarray] = {}
        self._semantic_index_path_cache: dict[str, Path] = {}
        self._semantic_index_cache: dict[str, faiss.Index] = {}
        self._encoder_cache: dict[str, SentenceTransformer] = {}
        self._document_scope_cache: dict[str, dict[str, Any]] = {}

    def retriever_kind(self, retriever_name: str) -> str:
        return "lexical" if is_lexical_retriever(self.resolve_retriever_name(retriever_name)) else "semantic"

    def enabled_lexical_retrievers(self) -> list[str]:
        return expand_retriever_names(
            self.config.enabled_lexical_retrievers,
            [],
            self.config.enabled_document_scopes,
            getattr(self.config, "full_scope_retrievers", None),
        )

    def enabled_semantic_retrievers(self) -> list[str]:
        return expand_retriever_names(
            [],
            self.config.enabled_semantic_retrievers,
            self.config.enabled_document_scopes,
            getattr(self.config, "full_scope_retrievers", None),
        )

    def enabled_retrievers(self) -> list[str]:
        return ordered_enabled_retrievers(
            self.config.enabled_lexical_retrievers,
            self.config.enabled_semantic_retrievers,
            self.config.enabled_document_scopes,
            getattr(self.config, "full_scope_retrievers", None),
        )

    def resolve_retriever_name(self, retriever_name: str) -> str:
        enabled = set(self.enabled_retrievers())
        if retriever_name in enabled:
            return retriever_name
        base_name, document_scope = split_retriever_name(retriever_name)
        if base_name in self.config.enabled_lexical_retrievers or base_name in self.config.enabled_semantic_retrievers:
            preferred = compose_retriever_name(base_name, document_scope)
            if preferred in enabled:
                return preferred
            if document_scope in self.config.enabled_document_scopes:
                return preferred
            if len(self.config.enabled_document_scopes) == 1:
                fallback = compose_retriever_name(base_name, self.config.enabled_document_scopes[0])
                if fallback in enabled:
                    return fallback
        raise KeyError(f"Unknown retriever: {retriever_name}")

    def paired_retriever(self, base_name: str, retriever_name: str) -> str:
        resolved = self.resolve_retriever_name(retriever_name)
        return compose_retriever_name(base_name, retriever_document_scope(resolved))

    def semantic_spec(self, retriever_name: str) -> SemanticRetrieverSpec:
        resolved = self.resolve_retriever_name(retriever_name)
        return self.config.semantic_retrievers[base_retriever_name(resolved)]

    def document_scope_bundle(self, retriever_name: str) -> dict[str, Any]:
        resolved = self.resolve_retriever_name(retriever_name)
        document_scope = retriever_document_scope(resolved)
        if document_scope in self._document_scope_cache:
            return self._document_scope_cache[document_scope]
        if document_scope == DEFAULT_DOCUMENT_SCOPE:
            bundle = {
                "texts": self.publications["document_text"].tolist(),
                "tokens": self.publications["document_tokens"].tolist(),
                "token_texts": self.publications["document_token_text"].tolist(),
            }
            self._document_scope_cache[document_scope] = bundle
            return bundle
        fields = DOCUMENT_SCOPE_FIELDS[document_scope]
        texts = [
            join_fields(*(str(row[field]) for field in fields))
            for row in self.publications[list(fields)].fillna("").to_dict(orient="records")
        ]
        tokens = [tokenize(text) for text in texts]
        bundle = {
            "texts": texts,
            "tokens": tokens,
            "token_texts": [" ".join(values) for values in tokens],
        }
        self._document_scope_cache[document_scope] = bundle
        return bundle

    def rank_queries(
        self,
        query_frame: pd.DataFrame,
        retriever_name: str,
        max_depth: int | None = None,
        query_embeddings: np.ndarray | None = None,
    ) -> pd.DataFrame:
        retriever_name = self.resolve_retriever_name(retriever_name)
        limit = int(max_depth or self.config.max_ranking_depth)
        if is_lexical_retriever(retriever_name):
            return self._rank_lexical_frame(query_frame, retriever_name, limit)
        return self._rank_semantic_frame(query_frame, retriever_name, limit, query_embeddings=query_embeddings)

    def encode_query_frame(
        self,
        query_frame: pd.DataFrame,
        retriever_name: str,
        label: str | None = None,
    ) -> np.ndarray:
        spec = self.semantic_spec(retriever_name)
        query_inputs = [(spec.query_prefix + text).strip() for text in query_frame["text"].tolist()]
        return self._encode_texts_in_memory(spec, query_inputs, label or f"queries for {base_retriever_name(self.resolve_retriever_name(retriever_name))}")

    def ensure_rankings(self, split_key: str, retriever_name: str) -> pd.DataFrame:
        retriever_name = self.resolve_retriever_name(retriever_name)
        cache_key = (split_key, retriever_name)
        if cache_key in self._ranking_cache:
            return self._ranking_cache[cache_key]
        ranking_path = self.store.path("rankings", split_key, f"{retriever_name}_top_{self.config.max_ranking_depth}.parquet")
        if ranking_path.exists():
            ranking = self.store.read_parquet(ranking_path)
            self._ranking_cache[cache_key] = ranking
            return ranking
        if is_lexical_retriever(retriever_name):
            rankings = self._rank_lexical(split_key, retriever_name)
        else:
            rankings = self._rank_semantic(split_key, retriever_name)
        self.store.write_parquet(ranking_path, rankings)
        self._ranking_cache[cache_key] = rankings
        return rankings

    def discard_cached_ranking(self, split_key: str, retriever_name: str) -> None:
        retriever_name = self.resolve_retriever_name(retriever_name)
        self._ranking_cache.pop((split_key, retriever_name), None)

    def ensure_all_rankings(self) -> None:
        for split_key in self.queries:
            for retriever_name in self.enabled_retrievers():
                self.console.log(f"Ranking {split_key} with {retriever_name}")
                self.ensure_rankings(split_key, retriever_name)

    def ensure_lexical_artifacts(self, retriever_name: str) -> LexicalArtifacts:
        retriever_name = self.resolve_retriever_name(retriever_name)
        if retriever_name in self._lexical_artifact_cache:
            return self._lexical_artifact_cache[retriever_name]
        base_name = base_retriever_name(retriever_name)
        shared = self.ensure_shared_lexical_artifacts(retriever_name)
        if base_name == "bm25":
            artifacts = LexicalArtifacts(
                name=retriever_name,
                model=self.ensure_bm25_model(retriever_name),
                vectorizer=None,
                matrix=None,
                count_vectorizer=shared.count_vectorizer,
                count_matrix=shared.count_matrix,
                binary_matrix=shared.binary_matrix,
                document_lengths=shared.document_lengths,
                vocabulary=shared.vocabulary,
            )
        elif base_name == "tfidf":
            tfidf_payload = self.ensure_tfidf_payload(retriever_name)
            artifacts = LexicalArtifacts(
                name=retriever_name,
                model=None,
                vectorizer=tfidf_payload["vectorizer"],
                matrix=tfidf_payload["matrix"],
                count_vectorizer=shared.count_vectorizer,
                count_matrix=shared.count_matrix,
                binary_matrix=shared.binary_matrix,
                document_lengths=shared.document_lengths,
                vocabulary=shared.vocabulary,
            )
        else:
            artifacts = LexicalArtifacts(
                name=retriever_name,
                model=None,
                vectorizer=shared.count_vectorizer,
                matrix=shared.binary_matrix,
                count_vectorizer=shared.count_vectorizer,
                count_matrix=shared.count_matrix,
                binary_matrix=shared.binary_matrix,
                document_lengths=shared.document_lengths,
                vocabulary=shared.vocabulary,
            )
        self._lexical_artifact_cache[retriever_name] = artifacts
        return artifacts

    def ensure_shared_lexical_artifacts(self, retriever_name: str = DEFAULT_DOCUMENT_SCOPE) -> SharedLexicalArtifacts:
        document_scope = retriever_name if retriever_name in DOCUMENT_SCOPE_FIELDS else retriever_document_scope(self.resolve_retriever_name(retriever_name))
        if document_scope in self._shared_lexical_cache:
            return self._shared_lexical_cache[document_scope]
        cache_path = self.store.path("indexes", "lexical", document_scope, "corpus.joblib")
        if cache_path.exists():
            payload = self.store.read_joblib(cache_path)
            self._shared_lexical_cache[document_scope] = SharedLexicalArtifacts(
                count_vectorizer=payload["count_vectorizer"],
                count_matrix=payload["count_matrix"],
                binary_matrix=payload["binary_matrix"],
                document_lengths=payload["document_lengths"],
                vocabulary=payload["vocabulary"],
            )
            return self._shared_lexical_cache[document_scope]
        self.console.log(f"Building shared lexical corpus artifacts for {document_scope}")
        document_token_text = self.document_scope_bundle(compose_retriever_name("bm25", document_scope))["token_texts"]
        count_vectorizer = CountVectorizer(
            tokenizer=str.split,
            token_pattern=None,
            preprocessor=None,
            lowercase=False,
            binary=False,
        )
        count_matrix = count_vectorizer.fit_transform(document_token_text).tocsr()
        binary_matrix = count_matrix.copy()
        binary_matrix.data = np.ones(binary_matrix.nnz, dtype=np.float32)
        payload = {
            "count_vectorizer": count_vectorizer,
            "count_matrix": count_matrix,
            "binary_matrix": binary_matrix,
            "document_lengths": np.asarray(count_matrix.sum(axis=1)).ravel().astype(np.int32),
            "vocabulary": count_vectorizer.vocabulary_,
        }
        self.store.write_joblib(cache_path, payload)
        self._shared_lexical_cache[document_scope] = SharedLexicalArtifacts(**payload)
        return self._shared_lexical_cache[document_scope]

    def ensure_tfidf_payload(self, retriever_name: str = DEFAULT_DOCUMENT_SCOPE) -> dict[str, Any]:
        document_scope = retriever_name if retriever_name in DOCUMENT_SCOPE_FIELDS else retriever_document_scope(self.resolve_retriever_name(retriever_name))
        if document_scope in self._tfidf_payload_cache:
            return self._tfidf_payload_cache[document_scope]
        cache_path = self.store.path("indexes", "lexical", document_scope, "tfidf.joblib")
        if cache_path.exists():
            self._tfidf_payload_cache[document_scope] = self.store.read_joblib(cache_path)
            return self._tfidf_payload_cache[document_scope]
        self.console.log(f"Building shared TF-IDF corpus artifacts for {document_scope}")
        document_token_text = self.document_scope_bundle(compose_retriever_name("tfidf", document_scope))["token_texts"]
        vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            token_pattern=None,
            preprocessor=None,
            lowercase=False,
            norm="l2",
            use_idf=True,
            smooth_idf=False,
            sublinear_tf=False,
        )
        matrix = vectorizer.fit_transform(document_token_text)
        self._tfidf_payload_cache[document_scope] = {"vectorizer": vectorizer, "matrix": matrix}
        self.store.write_joblib(cache_path, self._tfidf_payload_cache[document_scope])
        return self._tfidf_payload_cache[document_scope]

    def ensure_bm25_model(self, retriever_name: str = DEFAULT_DOCUMENT_SCOPE) -> Any:
        document_scope = retriever_name if retriever_name in DOCUMENT_SCOPE_FIELDS else retriever_document_scope(self.resolve_retriever_name(retriever_name))
        if document_scope in self._bm25_model_cache:
            return self._bm25_model_cache[document_scope]
        cache_path = self.store.path("indexes", "lexical", document_scope, "bm25.joblib")
        if cache_path.exists():
            self._bm25_model_cache[document_scope] = self.store.read_joblib(cache_path)
            return self._bm25_model_cache[document_scope]
        self.console.log(f"Building shared BM25 corpus artifacts for {document_scope}")
        document_tokens = self.document_scope_bundle(compose_retriever_name("bm25", document_scope))["tokens"]
        self._bm25_model_cache[document_scope] = BM25Okapi(document_tokens)
        self.store.write_joblib(cache_path, self._bm25_model_cache[document_scope])
        return self._bm25_model_cache[document_scope]

    def ensure_semantic_embeddings(self, retriever_name: str) -> tuple[SemanticRetrieverSpec, np.ndarray]:
        retriever_name = self.resolve_retriever_name(retriever_name)
        if retriever_name in self._semantic_embedding_cache:
            return self._semantic_embedding_cache[retriever_name]
        spec = self.semantic_spec(retriever_name)
        embedding_path = self.store.path("embeddings", retriever_name, "documents.npy")
        if embedding_path.exists():
            embeddings = self.store.read_numpy(embedding_path)
            self._semantic_embedding_cache[retriever_name] = (spec, embeddings)
            return spec, embeddings
        document_inputs = [
            (spec.document_prefix + text).strip()
            for text in self.document_scope_bundle(retriever_name)["texts"]
        ]
        embeddings = self._encode_texts(
            spec,
            document_inputs,
            f"collection documents for {retriever_name}",
            embedding_path,
            "documents_chunks",
        )
        self._semantic_embedding_cache[retriever_name] = (spec, embeddings)
        return spec, embeddings

    def ensure_query_embeddings(self, split_key: str, retriever_name: str) -> np.ndarray:
        retriever_name = self.resolve_retriever_name(retriever_name)
        base_name = base_retriever_name(retriever_name)
        cache_key = (split_key, base_name)
        if cache_key in self._query_embedding_cache:
            return self._query_embedding_cache[cache_key]
        embedding_path = self.store.path("embeddings", base_name, f"{split_key}.npy")
        if embedding_path.exists():
            embeddings = self.store.read_numpy(embedding_path)
            self._query_embedding_cache[cache_key] = embeddings
            return embeddings
        spec = self.semantic_spec(retriever_name)
        queries = self.queries[split_key]
        query_inputs = [(spec.query_prefix + text).strip() for text in queries["text"].tolist()]
        embeddings = self._encode_texts(
            spec,
            query_inputs,
            f"{split_key} queries for {base_name}",
            embedding_path,
            f"{split_key}_chunks",
        )
        self._query_embedding_cache[cache_key] = embeddings
        return embeddings

    def ensure_semantic_index(self, retriever_name: str) -> Path:
        retriever_name = self.resolve_retriever_name(retriever_name)
        if retriever_name in self._semantic_index_path_cache:
            return self._semantic_index_path_cache[retriever_name]
        index_path = self.store.path("indexes", "semantic", retriever_name, "documents.faiss")
        if index_path.exists():
            self._semantic_index_path_cache[retriever_name] = index_path
            return index_path
        _, embeddings = self.ensure_semantic_embeddings(retriever_name)
        self.console.log(f"Building FAISS index for {retriever_name}")
        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        index.add(embeddings)
        faiss.write_index(index, str(index_path))
        self._semantic_index_cache[retriever_name] = index
        self._semantic_index_path_cache[retriever_name] = index_path
        return index_path

    def ensure_semantic_index_object(self, retriever_name: str) -> faiss.Index:
        retriever_name = self.resolve_retriever_name(retriever_name)
        if retriever_name in self._semantic_index_cache:
            return self._semantic_index_cache[retriever_name]
        index_path = self.ensure_semantic_index(retriever_name)
        index = faiss.read_index(str(index_path))
        self._semantic_index_cache[retriever_name] = index
        return index

    def _rank_lexical(self, split_key: str, retriever_name: str) -> pd.DataFrame:
        retriever_name = self.resolve_retriever_name(retriever_name)
        queries = self.queries[split_key]
        if self.config.runtime.limit_queries is not None:
            queries = queries.head(self.config.runtime.limit_queries).copy()
        return self._rank_lexical_frame(queries, retriever_name, self.config.max_ranking_depth)

    def _rank_lexical_frame(self, query_frame: pd.DataFrame, retriever_name: str, max_depth: int) -> pd.DataFrame:
        retriever_name = self.resolve_retriever_name(retriever_name)
        artifacts = self.ensure_lexical_artifacts(retriever_name)
        rows: list[dict[str, Any]] = []
        depth = min(int(max_depth), len(self.document_ids))
        for query in query_frame.itertuples(index=False):
            scores = self._score_lexical_query(artifacts, list(query.query_tokens))
            order, top_scores = _top_k(scores, depth)
            for rank, (document_index, score) in enumerate(zip(order, top_scores, strict=True), start=1):
                rows.append(
                    {
                        "query_id": int(query.query_id),
                        "rank": rank,
                        "document_id": int(self.document_ids[int(document_index)]),
                        "score": float(score),
                    }
                )
        return pd.DataFrame(rows)

    def _rank_semantic(self, split_key: str, retriever_name: str) -> pd.DataFrame:
        retriever_name = self.resolve_retriever_name(retriever_name)
        queries = self.queries[split_key]
        if self.config.runtime.limit_queries is not None:
            queries = queries.head(self.config.runtime.limit_queries).copy()
        query_embeddings = self.ensure_query_embeddings(split_key, retriever_name)
        return self._rank_semantic_frame(queries, retriever_name, self.config.max_ranking_depth, query_embeddings=query_embeddings)

    def _rank_semantic_frame(
        self,
        query_frame: pd.DataFrame,
        retriever_name: str,
        max_depth: int,
        query_embeddings: np.ndarray | None = None,
    ) -> pd.DataFrame:
        retriever_name = self.resolve_retriever_name(retriever_name)
        index = self.ensure_semantic_index_object(retriever_name)
        embeddings = query_embeddings
        if embeddings is None:
            embeddings = self.encode_query_frame(query_frame, retriever_name, label=f"{retriever_name} queries")
        self.console.log(f"Searching FAISS with {retriever_name} for {len(query_frame)} queries")
        scores, indices = index.search(embeddings.astype(np.float32), min(int(max_depth), len(self.document_ids)))
        rows: list[dict[str, Any]] = []
        for query_id, query_scores, query_indices in zip(query_frame["query_id"], scores, indices, strict=True):
            for rank, (document_index, score) in enumerate(zip(query_indices, query_scores, strict=True), start=1):
                rows.append(
                    {
                        "query_id": int(query_id),
                        "rank": rank,
                        "document_id": int(self.document_ids[int(document_index)]),
                        "score": float(score),
                    }
                )
        return pd.DataFrame(rows)

    def _score_lexical_query(self, artifacts: LexicalArtifacts, tokens: list[str]) -> np.ndarray:
        base_name = base_retriever_name(artifacts.name)
        if base_name == "bm25":
            return np.asarray(artifacts.model.get_scores(tokens), dtype=np.float32)
        token_text = " ".join(tokens)
        query_vector = artifacts.vectorizer.transform([token_text])
        if base_name == "tfidf":
            return (artifacts.matrix @ query_vector.T).toarray().ravel().astype(np.float32)
        binary_query_vector = query_vector.copy()
        binary_query_vector.data = np.ones(binary_query_vector.nnz, dtype=np.float32)
        overlap = (artifacts.binary_matrix @ binary_query_vector.T).toarray().ravel().astype(np.float32)
        if base_name == "keyword_overlap":
            return overlap
        query_size = max(int(binary_query_vector.sum()), 1)
        denominator = artifacts.document_lengths + query_size - overlap
        denominator = np.where(denominator == 0, 1.0, denominator)
        return overlap / denominator

    def _resolve_encoder_model_name(self, model_name: str) -> str:
        candidate = Path(model_name)
        if candidate.is_absolute():
            return model_name
        root_dir = getattr(getattr(self, "config", None), "root_dir", None)
        if root_dir is None:
            return model_name
        repo_relative = (root_dir / candidate).resolve()
        if repo_relative.exists():
            return str(repo_relative)
        return model_name

    def _load_encoder(self, spec: SemanticRetrieverSpec) -> SentenceTransformer:
        if spec.name in self._encoder_cache:
            return self._encoder_cache[spec.name]
        resolved_model_name = self._resolve_encoder_model_name(spec.model_name)
        self.console.log(f"Loading sentence encoder {spec.name} ({resolved_model_name})")
        model_kwargs: dict[str, Any] | None = None
        if spec.code_revision is not None:
            model_kwargs = {"code_revision": spec.code_revision}
        model = SentenceTransformer(
            resolved_model_name,
            trust_remote_code=spec.trust_remote_code,
            revision=spec.revision,
            model_kwargs=model_kwargs,
            device="cpu",
        )
        self._encoder_cache[spec.name] = model
        return model

    def _encode_texts(
        self,
        spec: SemanticRetrieverSpec,
        texts: list[str],
        label: str,
        final_path: Path,
        chunk_dir_name: str,
    ) -> np.ndarray:
        if final_path.exists():
            return self.store.read_numpy(final_path)
        if not texts:
            embeddings = np.empty((0, 0), dtype=np.float32)
            self.store.write_numpy(final_path, embeddings)
            return embeddings
        chunk_size = self._embedding_chunk_size(spec)
        total_chunks = (len(texts) + chunk_size - 1) // chunk_size
        self.console.log(
            f"Encoding {label} with {spec.name} ({spec.model_name}) "
            f"[{len(texts)} items in {total_chunks} chunks of up to {chunk_size}]"
        )
        model = self._load_encoder(spec)
        chunk_dir = final_path.parent / chunk_dir_name
        chunk_dir.mkdir(parents=True, exist_ok=True)
        arrays: list[np.ndarray] = []
        cached_chunks = 0
        encoded_chunks = 0
        progress = tqdm(
            total=len(texts),
            desc=f"{spec.name} {label}",
            unit="item",
            **self.console.progress_kwargs(len(texts)),
        )
        try:
            for chunk_index, start in enumerate(range(0, len(texts), chunk_size)):
                end = min(start + chunk_size, len(texts))
                chunk_path = chunk_dir / f"chunk_{chunk_index:04d}_{start:06d}_{end - 1:06d}.npy"
                chunk_embeddings: np.ndarray
                if chunk_path.exists():
                    chunk_embeddings = self.store.read_numpy(chunk_path)
                    cached_chunks += 1
                else:
                    chunk_embeddings = model.encode(
                        texts[start:end],
                        batch_size=spec.batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=spec.normalize_embeddings,
                    ).astype(np.float32)
                    self.store.write_numpy(chunk_path, chunk_embeddings)
                    encoded_chunks += 1
                    gc.collect()
                arrays.append(chunk_embeddings)
                progress.update(end - start)
        finally:
            progress.close()
        embeddings = np.concatenate(arrays, axis=0)
        self.store.write_numpy(final_path, embeddings)
        self.console.log(
            f"Stored {label} embeddings for {spec.name}: "
            f"{cached_chunks} cached chunks, {encoded_chunks} new chunks"
        )
        return embeddings

    def _encode_texts_in_memory(self, spec: SemanticRetrieverSpec, texts: list[str], label: str) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        chunk_size = self._embedding_chunk_size(spec)
        total_chunks = (len(texts) + chunk_size - 1) // chunk_size
        self.console.log(
            f"Encoding {label} with {spec.name} ({spec.model_name}) "
            f"[{len(texts)} items in {total_chunks} chunks of up to {chunk_size}]"
        )
        model = self._load_encoder(spec)
        arrays: list[np.ndarray] = []
        progress = tqdm(
            total=len(texts),
            desc=f"{spec.name} {label}",
            unit="item",
            **self.console.progress_kwargs(len(texts)),
        )
        try:
            for start in range(0, len(texts), chunk_size):
                end = min(start + chunk_size, len(texts))
                chunk = model.encode(
                    texts[start:end],
                    batch_size=spec.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=spec.normalize_embeddings,
                ).astype(np.float32)
                arrays.append(chunk)
                progress.update(end - start)
        finally:
            progress.close()
        return np.concatenate(arrays, axis=0)

    @staticmethod
    def _embedding_chunk_size(spec: SemanticRetrieverSpec) -> int:
        return max(spec.batch_size, min(512, spec.batch_size * 16))


def load_rankings_by_query(frame: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {int(query_id): group.reset_index(drop=True) for query_id, group in frame.groupby("query_id", sort=False)}
