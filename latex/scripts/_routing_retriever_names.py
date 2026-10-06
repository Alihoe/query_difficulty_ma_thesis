from __future__ import annotations

ROUTING_RETRIEVER_DISPLAY_NAMES = {
    "bge_base_en_v1_5": "BGE-base-en-v1.5",
    "bge_large_en_v1_5": "BGE-large-en-v1.5",
    "bge_m3": "BGE-M3",
    "bm25": "BM25",
    "contriever_msmarco": "Contriever-MSMARCO",
    "e5_large_v2": "E5-large-v2",
    "gte_large_en_v1_5": "GTE-large-en-v1.5",
    "gtr_t5_xl": "GTR-T5-XL",
    "jaccard": "Jaccard",
    "keyword_overlap": "Exact-keyword matching",
    "multilingual_e5_base": "multilingual-E5-base",
    "multilingual_e5_large": "multilingual-E5-large",
    "qwen3_embedding_0_6b": "Qwen3-Embedding-0.6B",
    "tfidf": "TF-IDF",
}


def routing_retriever_display_name(name: str) -> str:
    if name not in ROUTING_RETRIEVER_DISPLAY_NAMES:
        raise KeyError(f"No display name for routing retriever {name!r}; add it to ROUTING_RETRIEVER_DISPLAY_NAMES")
    return ROUTING_RETRIEVER_DISPLAY_NAMES[name]
