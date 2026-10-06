from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import nltk
import spacy
from nltk.corpus import wordnet as wn
from spacy.lang.de.stop_words import STOP_WORDS as DE_STOPWORDS
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOPWORDS
from spacy.lang.fr.stop_words import STOP_WORDS as FR_STOPWORDS
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.utils import logging as transformers_logging

from query_difficulty.console import Console
from query_difficulty.dataset import TOKEN_PATTERN


STOPWORD_MAP = {
    "en": EN_STOPWORDS,
    "de": DE_STOPWORDS,
    "fr": FR_STOPWORDS,
}

SPACY_MODEL_MAP = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
}


@dataclass(slots=True)
class QueryLinguisticSignals:
    stopword_ratio: float
    proper_noun_ratio: float
    dependency_link_span: float
    mean_polysemy: float
    sentiment_polarity: float
    sentiment_intensity: float
    entities: set[str]


class TextServices:
    def __init__(self, allow_downloads: bool, console: Console) -> None:
        self.allow_downloads = allow_downloads
        self.console = console
        self._nlp_models: dict[str, spacy.language.Language] = {}
        self._sentiment = None
        self._logged_spacy_warnings: set[str] = set()

    @staticmethod
    def stopwords(language: str) -> set[str]:
        return set(STOPWORD_MAP.get(language, set()))

    def language_model(self, language: str) -> spacy.language.Language:
        if language in self._nlp_models:
            return self._nlp_models[language]
        model_name = SPACY_MODEL_MAP.get(language, "")
        if model_name:
            self.console.log(f"Loading spaCy model {model_name} for {language} query analysis")
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", category=UserWarning)
                    model = spacy.load(model_name, disable=[])
                self._log_spacy_load_warnings(model_name, caught)
            except Exception as error:
                raise RuntimeError(
                    f"Failed to load spaCy model {model_name} for language {language}. "
                    f"Install the model or enable downloads before generating features."
                ) from error
        else:
            raise RuntimeError(f"No spaCy model is configured for language {language}.")
        self._validate_model_capabilities(model, language, model_name)
        self._nlp_models[language] = model
        return model

    def analyze_query(self, text: str, language: str, tokens: list[str]) -> QueryLinguisticSignals:
        tokens = self._token_list(tokens)
        stopwords = self.stopwords(language)
        stopword_ratio = self._ratio(sum(1 for token in tokens if token in stopwords), len(tokens))
        try:
            doc = self.language_model(language)(text)
        except Exception as error:
            raise RuntimeError(f"spaCy query analysis failed for language {language}.") from error
        proper_noun_ratio = self._proper_noun_ratio(doc, text, tokens)
        dependency_link_span = self._dependency_link_span(doc)
        mean_polysemy = self._mean_polysemy(doc, language)
        sentiment_polarity, sentiment_intensity = self._sentiment_scores(text, language)
        entities = self._entities(doc)
        return QueryLinguisticSignals(
            stopword_ratio=stopword_ratio,
            proper_noun_ratio=proper_noun_ratio,
            dependency_link_span=dependency_link_span,
            mean_polysemy=mean_polysemy,
            sentiment_polarity=sentiment_polarity,
            sentiment_intensity=sentiment_intensity,
            entities=entities,
        )

    def analyze_queries(
        self,
        texts: list[str],
        language: str,
        token_lists: list[list[str]],
        batch_size: int = 64,
    ) -> list[QueryLinguisticSignals]:
        if not texts:
            return []
        docs = self._docs_for_texts(texts, language, batch_size=batch_size)
        sentiments = self._sentiment_scores_batch(texts, language, batch_size=batch_size)
        stopwords = self.stopwords(language)
        outputs = []
        for text, tokens, doc, sentiment in zip(texts, token_lists, docs, sentiments, strict=True):
            tokens = self._token_list(tokens)
            outputs.append(
                QueryLinguisticSignals(
                    stopword_ratio=self._ratio(sum(1 for token in tokens if token in stopwords), len(tokens)),
                    proper_noun_ratio=self._proper_noun_ratio(doc, text, tokens),
                    dependency_link_span=self._dependency_link_span(doc),
                    mean_polysemy=self._mean_polysemy(doc, language),
                    sentiment_polarity=sentiment[0],
                    sentiment_intensity=sentiment[1],
                    entities=self._entities(doc),
                )
            )
        return outputs

    def entity_set(self, text: str, language: str) -> set[str]:
        doc = self.language_model(language)(text)
        return self._entities(doc)

    def entity_sets(self, texts: list[str], language: str, batch_size: int = 128) -> list[set[str]]:
        if not texts:
            return []
        return [self._entities(doc) for doc in self._docs_for_texts(texts, language, batch_size=batch_size)]

    @staticmethod
    def _token_list(tokens: Any) -> list[str]:
        if tokens is None:
            return []
        if isinstance(tokens, list):
            return tokens
        if isinstance(tokens, tuple):
            return list(tokens)
        if isinstance(tokens, str):
            return [tokens]
        if hasattr(tokens, "tolist"):
            values = tokens.tolist()
            if isinstance(values, list):
                return values
            if isinstance(values, tuple):
                return list(values)
            return [str(values)] if values is not None else []
        return list(tokens)

    def _proper_noun_ratio(self, doc: spacy.tokens.Doc | None, text: str, tokens: list[str]) -> float:
        if len(tokens) == 0 and not text:
            return 0.0
        if doc is not None and doc.has_annotation("POS"):
            count, total = self._normalized_proper_noun_counts(doc)
            if total > 0:
                return self._ratio(count, total)
        raw_tokens = TOKEN_PATTERN.findall(text or "")
        heuristic = sum(1 for token in raw_tokens if token[:1].isalpha() and token[:1].isupper())
        return self._ratio(heuristic, len(raw_tokens))

    @staticmethod
    def _dependency_link_span(doc: spacy.tokens.Doc | None) -> float:
        if doc is None or not doc.has_annotation("DEP"):
            return math.nan
        spans = [
            abs(token.i - token.head.i)
            for token in doc
            if not token.is_punct and not token.is_space and token.i != token.head.i
        ]
        if not spans:
            return 0.0
        return float(sum(spans) / len(spans))

    def _mean_polysemy(self, doc: spacy.tokens.Doc | None, language: str) -> float:
        if language != "en":
            return math.nan
        if doc is None:
            return math.nan
        self._ensure_wordnet()
        values = []
        for token in doc:
            if not token.is_alpha:
                continue
            lemma = token.lemma_.lower() if token.lemma_ else token.text.lower()
            count = len(wn.synsets(lemma))
            values.append(float(count if count > 0 else 1))
        if not values:
            return math.nan
        return float(sum(values) / len(values))

    def _ensure_wordnet(self) -> None:
        try:
            wn.synsets("test")
            return
        except LookupError:
            if not self.allow_downloads:
                raise RuntimeError("NLTK WordNet data is required for English polysemy features, but downloads are disabled.")
            self.console.log("Downloading NLTK WordNet data")
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            try:
                wn.synsets("test")
                return
            except LookupError as error:
                raise RuntimeError("Failed to prepare NLTK WordNet data for English polysemy features.") from error

    def _sentiment_scores(self, text: str, language: str) -> tuple[float, float]:
        if language != "en":
            return math.nan, math.nan
        try:
            classifier = self._sentiment_pipeline()
        except Exception as error:
            raise RuntimeError("Failed to load the English sentiment model for query-signal generation.") from error
        try:
            output = classifier(text, top_k=None)[0]
        except Exception as error:
            raise RuntimeError("English sentiment inference failed during query-signal generation.") from error
        scores = {item["label"].lower(): float(item["score"]) for item in output}
        positive = scores.get("positive", 0.0)
        negative = scores.get("negative", 0.0)
        neutral = scores.get("neutral", 0.0)
        return positive - negative, 1.0 - neutral

    def _sentiment_scores_batch(self, texts: list[str], language: str, batch_size: int = 64) -> list[tuple[float, float]]:
        if language != "en":
            return [(math.nan, math.nan) for _ in texts]
        try:
            classifier = self._sentiment_pipeline()
        except Exception as error:
            raise RuntimeError("Failed to load the English sentiment model for batched query-signal generation.") from error
        outputs: list[tuple[float, float]] = []
        try:
            for start in range(0, len(texts), batch_size):
                end = min(start + batch_size, len(texts))
                batch_output = classifier(texts[start:end], top_k=None, batch_size=batch_size)
                if batch_output and isinstance(batch_output[0], dict):
                    batch_output = [batch_output]
                for item in batch_output:
                    labels = item if isinstance(item, list) else [item]
                    scores = {entry["label"].lower(): float(entry["score"]) for entry in labels}
                    positive = scores.get("positive", 0.0)
                    negative = scores.get("negative", 0.0)
                    neutral = scores.get("neutral", 0.0)
                    outputs.append((positive - negative, 1.0 - neutral))
            return outputs
        except Exception as error:
            raise RuntimeError("English sentiment inference failed during batched query-signal generation.") from error

    def _sentiment_pipeline(self):
        if self._sentiment is None:
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            mode_text = "with downloads enabled" if self.allow_downloads else "from the local cache only"
            self.console.log(f"Loading English sentiment model {model_name} {mode_text}")
            verbosity = transformers_logging.get_verbosity()
            try:
                transformers_logging.set_verbosity_error()
                if self.allow_downloads:
                    self._sentiment = pipeline(
                        "text-classification",
                        model=model_name,
                        tokenizer=model_name,
                        top_k=None,
                        device=-1,
                    )
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
                    self._sentiment = pipeline(
                        "text-classification",
                        model=model,
                        tokenizer=tokenizer,
                        top_k=None,
                        device=-1,
                    )
            finally:
                transformers_logging.set_verbosity(verbosity)
        return self._sentiment

    def _log_spacy_load_warnings(self, model_name: str, caught: list[warnings.WarningMessage]) -> None:
        for warning in caught:
            message = str(warning.message)
            if "[W095]" in message and model_name not in self._logged_spacy_warnings:
                self.console.log(
                    f"spaCy model {model_name} was trained for a different spaCy patch version; continuing with the installed model."
                )
                self._logged_spacy_warnings.add(model_name)

    def _docs_for_texts(self, texts: list[str], language: str, batch_size: int) -> list[spacy.tokens.Doc | None]:
        if not texts:
            return []
        try:
            model = self.language_model(language)
            return list(model.pipe(texts, batch_size=batch_size))
        except Exception as error:
            raise RuntimeError(
                f"spaCy batch analysis failed for language {language} while processing {len(texts)} texts."
            ) from error

    @staticmethod
    def _entities(doc: spacy.tokens.Doc | None) -> set[str]:
        if doc is None or not doc.ents:
            return set()
        return {ent.text.strip().lower() for ent in doc.ents if ent.text.strip()}

    @staticmethod
    def _normalized_proper_noun_counts(doc: spacy.tokens.Doc) -> tuple[int, int]:
        proper_noun_count = 0
        normalized_token_count = 0
        for token in doc:
            if token.is_space:
                continue
            normalized_parts = [part.lower() for part in TOKEN_PATTERN.findall(token.text or "")]
            if not normalized_parts:
                continue
            normalized_token_count += len(normalized_parts)
            if token.pos_ == "PROPN":
                proper_noun_count += len(normalized_parts)
        return proper_noun_count, normalized_token_count

    def _validate_model_capabilities(
        self,
        model: spacy.language.Language,
        language: str,
        model_name: str,
    ) -> None:
        try:
            probe = model("OpenAI tests the parser.")
        except Exception as error:
            raise RuntimeError(f"spaCy model {model_name} for {language} could not process a probe document.") from error
        if not probe.has_annotation("POS"):
            raise RuntimeError(f"spaCy model {model_name} for {language} does not provide POS annotations.")
        if not probe.has_annotation("DEP"):
            raise RuntimeError(f"spaCy model {model_name} for {language} does not provide dependency annotations.")
        if "ner" not in model.pipe_names:
            raise RuntimeError(f"spaCy model {model_name} for {language} does not provide NER annotations.")

    @staticmethod
    def _ratio(count: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return float(count) / float(total)
