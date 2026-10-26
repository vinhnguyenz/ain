"""Optional tokenizer / representation helpers — infrastructure only.

These functions turn raw ticket text into tokens or a bag-of-words count
vector. They are provided so you can skip repetitive string-cleanup code;
using them is optional and you are free to replace them with your own
tokenization, n-grams, or a library vectorizer (e.g.
``sklearn.feature_extraction.text.CountVectorizer`` / ``TfidfVectorizer``).

Nothing in this file trains, fits, or predicts anything. The graded AI
core (representation choice, model training, and prediction) belongs in
``starter/student_core.py``.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, and split on whitespace/punctuation.

    Example: "The VPN isn't working!" -> ["the", "vpn", "isn", "t", "working"]
    """
    return _TOKEN_RE.findall(text.lower())


def build_vocabulary(texts: Iterable[str], min_freq: int = 1) -> List[str]:
    """Return a sorted vocabulary of tokens appearing at least ``min_freq``
    times across ``texts``. Deterministic (sorted), so vocabulary order
    does not depend on set/dict iteration order."""
    counts: Counter = Counter()
    for text in texts:
        counts.update(tokenize(text))
    return sorted(token for token, count in counts.items() if count >= min_freq)


def vectorize(text: str, vocabulary: List[str]) -> List[int]:
    """Return a bag-of-words count vector for ``text`` over ``vocabulary``.

    The vector has one entry per vocabulary word, in the same order as
    ``vocabulary``, counting how many times that word appears in ``text``.
    """
    token_counts = Counter(tokenize(text))
    return [token_counts.get(word, 0) for word in vocabulary]


def vocabulary_index(vocabulary: List[str]) -> Dict[str, int]:
    """Return a ``{word: column_index}`` mapping for ``vocabulary``."""
    return {word: idx for idx, word in enumerate(vocabulary)}
