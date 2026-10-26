"""Sanity tests for Project 10 (Helpdesk Ticket Router).

These tests check that the synthetic data loads, matches its schema and
label set, is reproducible from the generator, and that the graded AI
core (``starter/student_core.py``) is still empty. They do not grade a
student's classifier.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_module(unique_name: str, path: Path):
    """Load a module from an explicit file path under a project-unique
    name. Project 9 and Project 10 both ship a `starter` package; loading
    by path (instead of `sys.path` + `import starter`) avoids a module-name
    collision when both projects' tests run in the same pytest session."""
    spec = importlib.util.spec_from_file_location(unique_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module  # required for dataclasses' postponed-annotation lookup
    spec.loader.exec_module(module)
    return module


student_core = _load_module("project10_starter_student_core", PROJECT_ROOT / "starter" / "student_core.py")
_tokenizer = _load_module("project10_starter_tokenizer", PROJECT_ROOT / "starter" / "tokenizer.py")
build_vocabulary = _tokenizer.build_vocabulary
tokenize = _tokenizer.tokenize
vectorize = _tokenizer.vectorize

DATA_DIR = PROJECT_ROOT / "data"
EXPECTED_LABELS = {"Account", "Network", "Hardware", "Software", "Security", "Other"}
SPLIT_FILES = ["train.csv", "validation.csv", "test.csv", "hard_cases.csv"]


def _read_csv(path: Path):
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module", autouse=True)
def ensure_data_exists():
    if not (DATA_DIR / "train.csv").exists():
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "generate_tickets.py"), "--seed", "42"],
            check=True,
            cwd=PROJECT_ROOT,
        )


@pytest.mark.parametrize("filename", SPLIT_FILES)
def test_csv_schema_is_exact(filename):
    rows = _read_csv(DATA_DIR / filename)
    assert rows, f"{filename} must not be empty"
    assert list(rows[0].keys()) == ["ticket_id", "text", "label"]


@pytest.mark.parametrize("filename", SPLIT_FILES)
def test_labels_are_exactly_the_six_classes(filename):
    rows = _read_csv(DATA_DIR / filename)
    labels = {row["label"] for row in rows}
    assert labels.issubset(EXPECTED_LABELS), f"{filename} has unexpected labels: {labels - EXPECTED_LABELS}"


@pytest.mark.parametrize("filename", SPLIT_FILES)
def test_ticket_ids_are_unique(filename):
    rows = _read_csv(DATA_DIR / filename)
    ids = [row["ticket_id"] for row in rows]
    assert len(ids) == len(set(ids)), f"{filename} has duplicate ticket_id values"


def test_total_ticket_count_in_range():
    total = sum(len(_read_csv(DATA_DIR / name)) for name in ["train.csv", "validation.csv", "test.csv"])
    assert 1500 <= total <= 5000, f"train+validation+test must be 1500-5000 tickets, got {total}"


def test_label_distribution_is_not_extremely_skewed():
    rows = _read_csv(DATA_DIR / "train.csv")
    counts = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    assert set(counts.keys()) == EXPECTED_LABELS, "train.csv must use all six labels"
    ratio = max(counts.values()) / min(counts.values())
    assert ratio < 3.0, f"train.csv label distribution is too skewed: {counts}"


def test_hard_cases_has_at_least_one_case_per_class():
    rows = _read_csv(DATA_DIR / "hard_cases.csv")
    labels = {row["label"] for row in rows}
    assert labels == EXPECTED_LABELS, f"hard_cases.csv must cover all six labels, got {labels}"
    assert len(rows) >= 12, "hard_cases.csv should have a meaningful robustness sample"


def test_sha256sums_matches_committed_data_files():
    sums_path = DATA_DIR / "SHA256SUMS"
    assert sums_path.exists(), "data/SHA256SUMS must be committed alongside the CSVs"
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        expected_digest, filename = line.split(maxsplit=1)
        actual_digest = hashlib.sha256((DATA_DIR / filename).read_bytes()).hexdigest()
        assert actual_digest == expected_digest, f"{filename} does not match SHA256SUMS"


def test_regenerating_with_seed_42_reproduces_sha256sums(tmp_path):
    """Re-running the generator with --seed 42 must reproduce the exact
    bytes already committed in data/SHA256SUMS (schema + reward-design
    style determinism, no hidden timestamp/randomness)."""
    out_dir = tmp_path / "regenerated"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_tickets.py"),
            "--seed",
            "42",
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    committed_sums = (DATA_DIR / "SHA256SUMS").read_text(encoding="utf-8")
    regenerated_sums = (out_dir / "SHA256SUMS").read_text(encoding="utf-8")
    assert regenerated_sums == committed_sums


def test_label_description_documents_all_classes():
    text = (DATA_DIR / "label_description.md").read_text(encoding="utf-8")
    for label in EXPECTED_LABELS:
        assert label in text, f"label_description.md must mention '{label}'"


# -- tokenizer helper sanity (infrastructure, not the graded core) ----------


def test_tokenizer_lowercases_and_strips_punctuation():
    assert tokenize("The VPN isn't working!") == ["the", "vpn", "isn", "t", "working"]


def test_vectorize_matches_vocabulary_length():
    vocab = build_vocabulary(["hello world", "hello there"])
    vector = vectorize("hello hello world", vocab)
    assert len(vector) == len(vocab)
    assert sum(vector) == 3


# -- forbidden-core guards ---------------------------------------------------


def test_train_model_is_not_implemented():
    with pytest.raises(NotImplementedError):
        student_core.train_model(["some ticket text"], ["Account"])


def test_predict_is_not_implemented():
    with pytest.raises(NotImplementedError):
        student_core.predict(None, ["some ticket text"])


def test_no_shipped_model_artifacts_in_project():
    forbidden_suffixes = (".pkl", ".pt", ".joblib")
    offenders = [
        str(p)
        for p in PROJECT_ROOT.rglob("*")
        if p.is_file() and (p.suffix in forbidden_suffixes or p.name == "qtable.json")
    ]
    assert not offenders, f"No fitted model artifacts may ship in the starter tree: {offenders}"


def test_no_pii_patterns_in_ticket_data():
    """Emails at hanu/gmail domains, or 9+ digit runs (phone/ID-like),
    must not appear in the ticket text or label columns."""
    import re

    pii_pattern = re.compile(r"@(hanu|gmail)\.|[0-9]{9,}")
    for filename in SPLIT_FILES:
        rows = _read_csv(DATA_DIR / filename)
        for row in rows:
            assert not pii_pattern.search(row["text"]), f"PII-like pattern found in {filename}: {row['text']}"
