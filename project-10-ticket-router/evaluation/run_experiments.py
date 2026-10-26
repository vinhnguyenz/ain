#!/usr/bin/env python3
"""Midterm controlled experiment — Project 10 (IT Helpdesk Ticket Router).

Trains every pipeline variant defined in ``starter/student_core.py`` on
``data/train.csv`` and evaluates, for each variant:

* validation + test: Accuracy, macro Precision / Recall / F1
  (model selection is done on VALIDATION; test is reported once)
* a multi-class confusion matrix per split (saved as PNG)
* robustness on ``data/hard_cases.csv``: overall accuracy, degradation
  vs. the same model's test accuracy, and a per-category breakdown
  (typo / paraphrase / irrelevant-filler / ambiguous, following the ID
  scheme documented in ``data/label_description.md``)
* the test-set misclassifications of the best variant (for the
  qualitative error analysis in ``results.md``)

Outputs are written next to this script: ``results.md`` + ``cm_*.png``.

Usage:
    python evaluation/run_experiments.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from starter import student_core  # noqa: E402

DATA = PROJECT_ROOT / "data"
OUT = Path(__file__).resolve().parent
LABELS = list(student_core.LABELS)


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


def macro_metrics(y_true, y_pred) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def save_cm(y_true, y_pred, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="left")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if cm[i, j]:
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=9,
                )
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def hard_category(ticket_id: str) -> str:
    """Category of a hard case, per data/label_description.md."""
    n = int(ticket_id.split("-")[1])
    if n >= 19:
        return "ambiguous"
    return ["typo", "paraphrase", "irrelevant"][(n - 1) % 3]


def main() -> None:
    train = load("train")
    val = load("validation")
    test = load("test")
    hard = load("hard_cases")
    hard["category"] = hard["ticket_id"].map(hard_category)

    rows = []
    hard_rows = []
    per_variant = {}

    for variant in student_core.VARIANTS:
        model = student_core.train_model(
            list(train["text"]), list(train["label"]), variant=variant
        )
        entry = {"variant": variant}
        for split_name, df in [("validation", val), ("test", test)]:
            pred = student_core.predict(model, list(df["text"]))
            m = macro_metrics(list(df["label"]), pred)
            entry[split_name] = m
            save_cm(
                list(df["label"]),
                pred,
                f"{variant} — {split_name}",
                OUT / f"cm_{variant}_{split_name}.png",
            )
            if variant == student_core.DEFAULT_VARIANT and split_name == "test":
                per_variant["test_pred"] = pred
        pred_hard = student_core.predict(model, list(hard["text"]))
        hard_acc = accuracy_score(list(hard["label"]), pred_hard)
        entry["hard_acc"] = hard_acc
        entry["hard_drop"] = entry["test"]["accuracy"] - hard_acc
        hard_pred_map = dict(zip(hard["ticket_id"], pred_hard))
        for cat, grp in hard.groupby("category"):
            cat_pred = [hard_pred_map[t] for t in grp["ticket_id"]]
            hard_rows.append(
                {
                    "variant": variant,
                    "category": cat,
                    "n": len(grp),
                    "accuracy": accuracy_score(list(grp["label"]), cat_pred),
                }
            )
        per_variant[variant] = entry
        rows.append(entry)
        print(
            f"{variant:14s} val F1={entry['validation']['f1']:.4f}  "
            f"test F1={entry['test']['f1']:.4f}  test acc={entry['test']['accuracy']:.4f}  "
            f"hard acc={hard_acc:.4f} (drop {entry['hard_drop']:+.4f})"
        )

    # Model selection: validation macro-F1 first; hard-case robustness
    # breaks ties (all variants tie at 1.0000 on validation here).
    best = max(rows, key=lambda e: (e["validation"]["f1"], e["hard_acc"]))
    print(
        f"\nBest (validation macro-F1, tie-break = hard-case accuracy): "
        f"{best['variant']}"
    )

    # ---- error analysis data for the best variant -------------------------
    model = student_core.train_model(
        list(train["text"]), list(train["label"]), variant=best["variant"]
    )
    test_pred = student_core.predict(model, list(test["text"]))
    errors = test.copy()
    errors["predicted"] = test_pred
    errors = errors[errors["label"] != errors["predicted"]]
    print(f"Best variant test errors: {len(errors)} / {len(test)}")

    hard_pred = student_core.predict(model, list(hard["text"]))
    hard_out = hard.copy()
    hard_out["predicted"] = hard_pred
    hard_out["correct"] = hard_out["label"] == hard_out["predicted"]

    # ---- split-overlap diagnostics (why are val/test scores perfect?) -----
    from starter.tokenizer import tokenize as _tok

    train_texts = set(train["text"])
    dup_rates = {
        name: float(df["text"].isin(train_texts).mean())
        for name, df in [("validation", val), ("test", test)]
    }
    train_tok = [(set(_tok(t)), lab) for t, lab in zip(train["text"], train["label"])]
    sample = test.sample(60, random_state=0)
    jacc = []
    for _, r in sample.iterrows():
        tt = set(_tok(r["text"]))
        jacc.append(
            max(
                (len(tt & s) / max(1, len(tt | s)) for s, lab in train_tok if lab == r["label"]),
                default=0.0,
            )
        )
    mean_jacc = float(np.mean(jacc)) if jacc else 0.0
    print(
        f"Overlap: exact-duplicate rate val={dup_rates['validation']:.3f} "
        f"test={dup_rates['test']:.3f}; mean best-Jaccard(test, train|same label)={mean_jacc:.3f}"
    )

    # ---- data-efficiency probe (qualitative error analysis material) ------
    sub = train.sample(frac=0.10, random_state=7)
    probe_model = student_core.train_model(
        list(sub["text"]), list(sub["label"]), variant=best["variant"]
    )
    probe_pred = student_core.predict(probe_model, list(test["text"]))
    probe_acc = accuracy_score(list(test["label"]), probe_pred)
    probe_errors = test.copy()
    probe_errors["predicted"] = probe_pred
    probe_errors = probe_errors[probe_errors["label"] != probe_errors["predicted"]]
    print(
        f"Probe (10% of train, n={len(sub)}): test acc={probe_acc:.4f}, "
        f"errors={len(probe_errors)}"
    )

    # ---- write results.md --------------------------------------------------
    lines = []
    lines.append("# Midterm experiment results — Project 10 (Ticket Router)\n")
    lines.append(
        "All numbers produced by `evaluation/run_experiments.py` "
        f"(train n={len(train)}, validation n={len(val)}, test n={len(test)}, "
        f"hard_cases n={len(hard)}). Model selection used the **validation** "
        "split; test metrics are reported once, for the selected pipeline.\n"
    )
    lines.append("## 1. Pipeline comparison (macro-averaged)\n")
    lines.append("| Variant | Split | Accuracy | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|")
    for e in rows:
        for split in ["validation", "test"]:
            m = e[split]
            lines.append(
                f"| {e['variant']} | {split} | {m['accuracy']:.4f} | "
                f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |"
            )
    lines.append(
        f"\nSelected pipeline (validation macro-F1, tie-break = hard-case "
        f"robustness): **{best['variant']}**.\n"
    )
    lines.append("## 1b. Why are validation/test scores perfect? (split-overlap check)\n")
    lines.append(
        "`label_description.md` warns that 100% F1 should trigger a leakage "
        "check. Our pipelines fit **only** on the training split (vectorizer "
        "vocabulary and classifier are learned from train.csv alone), so there "
        "is no fitting leakage. The perfect scores come from the split itself: "
        "the corpus is template-generated and split randomly, so near-identical "
        "template siblings land in both train and test.\n"
    )
    lines.append(
        f"- Exact-duplicate rate vs train: validation "
        f"{dup_rates['validation']:.1%}, test {dup_rates['test']:.1%} of texts "
        "appear verbatim in train.csv.\n"
        f"- Mean best Jaccard similarity between a test ticket's token set and "
        f"a same-label train ticket: {mean_jacc:.3f} (sample of 60).\n"
    )
    lines.append(
        "Conclusion: validation/test measure template memorisation more than "
        "generalisation here; `hard_cases.csv` and the ambiguous cases are the "
        "discriminative evaluation, and the data-efficiency probe in section 4 "
        "shows how the selected pipeline behaves when memorisation is limited.\n"
    )
    lines.append("## 2. Robustness on hard_cases.csv\n")
    lines.append("| Variant | Hard accuracy | Test accuracy | Degradation |")
    lines.append("|---|---|---|---|")
    for e in rows:
        lines.append(
            f"| {e['variant']} | {e['hard_acc']:.4f} | "
            f"{e['test']['accuracy']:.4f} | {e['hard_drop']:+.4f} |"
        )
    lines.append("\nPer-category accuracy (hard cases):\n")
    lines.append("| Variant | Category | n | Accuracy |")
    lines.append("|---|---|---|---|")
    cat_order = {"typo": 0, "paraphrase": 1, "irrelevant": 2, "ambiguous": 3}
    for r in sorted(hard_rows, key=lambda r: (r["variant"], cat_order[r["category"]])):
        lines.append(
            f"| {r['variant']} | {r['category']} | {r['n']} | {r['accuracy']:.4f} |"
        )
    lines.append("\nHard-case detail for the selected pipeline:\n")
    lines.append("| ID | Category | True | Predicted | Correct | Text |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in hard_out.iterrows():
        text = str(r["text"]).replace("|", "/")[:130]
        lines.append(
            f"| {r['ticket_id']} | {r['category']} | {r['label']} | "
            f"{r['predicted']} | {'yes' if r['correct'] else 'NO'} | {text} |"
        )
    lines.append("\n## 3. Test-set misclassifications of the selected pipeline\n")
    lines.append(f"Total errors: {len(errors)} / {len(test)}.\n")
    lines.append("| ID | True | Predicted | Text |")
    lines.append("|---|---|---|---|")
    for _, r in errors.iterrows():
        text = str(r["text"]).replace("|", "/")[:160]
        lines.append(
            f"| {r['ticket_id']} | {r['label']} | {r['predicted']} | {text} |"
        )
    lines.append(
        "\nConfusion matrices are saved as `cm_<variant>_<split>.png` "
        "next to this file. With zero errors on validation/test they are "
        "perfectly diagonal; the informative matrix is the hard-case one "
        "implied by section 2.\n"
    )
    lines.append("## 4. Data-efficiency probe (10% of train) — error-analysis material\n")
    lines.append(
        f"Selected pipeline retrained on a random 10% subsample of train "
        f"(n={len(sub)}, seed 7): test accuracy {probe_acc:.4f} with "
        f"{len(probe_errors)} errors (vs 0 errors with the full training set). "
        "These misclassifications are analysed qualitatively in "
        "`error_analysis.md`.\n"
    )
    lines.append("| ID | True | Predicted | Text |")
    lines.append("|---|---|---|---|")
    for _, r in probe_errors.iterrows():
        text = str(r["text"]).replace("|", "/")[:160]
        lines.append(
            f"| {r['ticket_id']} | {r['label']} | {r['predicted']} | {text} |"
        )
    lines.append(
        "\n### Per-class report (selected pipeline, test split)\n\n```\n"
        + classification_report(
            list(test["label"]), test_pred, labels=LABELS, zero_division=0
        )
        + "```\n"
    )
    (OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT / 'results.md'} and confusion-matrix PNGs.")


if __name__ == "__main__":
    main()
