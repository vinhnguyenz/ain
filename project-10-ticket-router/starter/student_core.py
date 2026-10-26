"""Student AI core — Project 10 (IT Helpdesk Ticket Router).

Implemented for the Midterm milestone (AI Core).

Design
------
The router is a classic supervised text-classification pipeline built
with scikit-learn:

    raw text
      -> tokenization (the course-provided ``tokenizer.tokenize``:
         lowercase + ``[a-z0-9]+`` tokens)
      -> feature extraction (Bag-of-Words counts or TF-IDF, word
         n-grams, optional character n-grams for typo robustness)
      -> classifier (Multinomial Naive Bayes, Logistic Regression, or
         a calibrated Linear SVM)

Three named pipeline variants are provided so the midterm controlled
experiment can compare them through one entry point:

    ``bow_nb``        CountVectorizer (BoW, uni+bi-grams) + MultinomialNB
    ``tfidf_svm``     TfidfVectorizer (uni+bi-grams) + LinearSVC
                      (wrapped in CalibratedClassifierCV so the UI can
                      show calibrated confidence scores)
    ``tfidf_logreg``  FeatureUnion of word-level TF-IDF (uni+bi-grams)
                      and character-level TF-IDF (char_wb 3-5 grams)
                      + Logistic Regression. The character features make
                      the model degrade gracefully on misspelled words
                      (e.g. "passowrd" still overlaps "password").

Contract (as used by ``starter/app.py`` and the sanity tests):
    ``train_model(texts, labels)`` -> trained model object
    ``predict(model, texts)``      -> list of labels, one per input text

``predict_proba`` is an additional helper (not part of the required
contract) that exposes per-class confidence scores for the Streamlit
UI. Empty / whitespace-only tickets carry no lexical signal; they are
routed to ``Other`` by an explicit, documented fallback in both
``predict`` and ``predict_proba``.

No pre-fitted model is shipped, no external API is called, and no
ticket text is hard-coded anywhere in this file.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

try:  # Prefer the course-provided tokenizer when the package is importable.
    from starter.tokenizer import tokenize as _course_tokenize
except Exception:  # pragma: no cover - fallback when loaded by file path
    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def _course_tokenize(text: str) -> List[str]:
        return _TOKEN_RE.findall(text.lower())


LABELS: Sequence[str] = ("Account", "Network", "Hardware", "Software", "Security", "Other")

#: Fallback queue for degenerate inputs (empty / whitespace-only text).
EMPTY_INPUT_LABEL = "Other"


def _word_vectorizer(vectorizer_cls, **overrides):
    """Word-level vectorizer using the course tokenizer.

    ``lowercase=False`` + ``token_pattern=None`` are required when a
    custom tokenizer is supplied; lowercasing already happens inside
    ``tokenize``. English stop words are removed so class-neutral filler
    sentences (greetings, sign-offs, out-of-office notes) contribute as
    little signal as possible.
    """
    params = dict(
        tokenizer=_course_tokenize,
        lowercase=False,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
    )
    params.update(overrides)
    return vectorizer_cls(**params)


def _build_pipeline(variant: str) -> Pipeline:
    if variant == "bow_nb":
        return Pipeline(
            [
                ("bow", _word_vectorizer(CountVectorizer)),
                ("clf", MultinomialNB(alpha=0.1)),
            ]
        )
    if variant == "tfidf_svm":
        return Pipeline(
            [
                ("tfidf", _word_vectorizer(TfidfVectorizer, sublinear_tf=True)),
                ("clf", CalibratedClassifierCV(LinearSVC(C=1.0), cv=3)),
            ]
        )
    if variant == "tfidf_logreg":
        features = FeatureUnion(
            [
                ("word", _word_vectorizer(TfidfVectorizer, sublinear_tf=True)),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        return Pipeline(
            [
                ("features", features),
                ("clf", LogisticRegression(C=10.0, max_iter=1000)),
            ]
        )
    raise ValueError(
        f"Unknown variant {variant!r}; expected one of {sorted(VARIANTS)}"
    )


#: Pipeline variants compared in the midterm controlled experiment.
VARIANTS: Sequence[str] = ("bow_nb", "tfidf_svm", "tfidf_logreg")

#: Variant used when ``train_model`` is called without one (the app and
#: the demo use this). In the midterm experiment all three variants tie
#: at validation macro-F1 = 1.0000, so the tie is broken by robustness
#: on ``data/hard_cases.csv``, where ``tfidf_svm`` degrades least
#: (accuracy 0.9167 vs 0.8750 / 0.8333) — see ``evaluation/results.md``.
DEFAULT_VARIANT = "tfidf_svm"


def train_model(
    texts: List[str], labels: List[str], variant: Optional[str] = None
) -> Any:
    """Train a ticket classifier on ``texts`` / ``labels``.

    Args:
        texts: raw ticket text, one string per training example.
        labels: the matching label for each text, one of ``LABELS``.
        variant: one of ``VARIANTS``; defaults to ``DEFAULT_VARIANT``.

    Returns:
        A fitted ``sklearn`` ``Pipeline`` (vectorizer + classifier).
    """
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    model = _build_pipeline(variant or DEFAULT_VARIANT)
    model.fit(list(texts), list(labels))
    return model


def _is_degenerate(text: Any) -> bool:
    return text is None or not str(text).strip()


def predict(model: Any, texts: List[str]) -> List[str]:
    """Predict a label from ``LABELS`` for each string in ``texts``.

    Returns one label per input text, in the same order. Empty or
    whitespace-only inputs are routed to ``Other`` (documented
    fallback); every other input is scored by the trained model.
    """
    texts = list(texts)
    results: List[str] = [""] * len(texts)
    real_idx = [i for i, t in enumerate(texts) if not _is_degenerate(t)]
    for i, t in enumerate(texts):
        if _is_degenerate(t):
            results[i] = EMPTY_INPUT_LABEL
    if real_idx:
        preds = model.predict([str(texts[i]) for i in real_idx])
        for i, p in zip(real_idx, preds):
            results[i] = str(p)
    return results


def predict_proba(model: Any, texts: List[str]) -> List[Dict[str, float]]:
    """Per-class confidence scores for each text (UI helper).

    Returns a list of ``{label: probability}`` dicts covering all six
    labels. Models are required to expose ``predict_proba`` (all three
    variants do — the SVM is probability-calibrated). Degenerate inputs
    mirror ``predict``'s fallback: ``Other`` with probability 1.0.
    """
    texts = list(texts)
    out: List[Dict[str, float]] = []
    real = [t for t in texts if not _is_degenerate(t)]
    proba_rows = []
    if real:
        proba_rows = list(model.predict_proba([str(t) for t in real]))
        classes = [str(c) for c in model.classes_]
    else:
        classes = []
    it = iter(proba_rows)
    for t in texts:
        if _is_degenerate(t):
            row = {label: 0.0 for label in LABELS}
            row[EMPTY_INPUT_LABEL] = 1.0
            out.append(row)
        else:
            probs = next(it)
            out.append(
                {label: float(probs[classes.index(label)]) for label in LABELS}
            )
    return out
