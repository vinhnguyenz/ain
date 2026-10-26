#!/usr/bin/env python3
"""Generate the synthetic English helpdesk-ticket corpus for Project 10.

This script is infrastructure only: it builds train/validation/test
splits plus a small robustness set (``hard_cases.csv``) from
template-based sentence generation. It does not train, fit, or ship any
classifier — that is student work (``starter/student_core.py``).

Usage
-----
    python3.11 scripts/generate_tickets.py --seed 42
    python3.11 scripts/generate_tickets.py --seed 2026 --out-dir /tmp/fixtures

Re-running with the same ``--seed`` and default ``--total`` must produce
byte-identical CSV files (verified against ``data/SHA256SUMS``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "data"

CLASSES: Tuple[str, ...] = ("Account", "Network", "Hardware", "Software", "Security", "Other")

# Not extremely skewed: every class gets between 15% and 20% of the corpus.
CLASS_WEIGHTS: Dict[str, float] = {
    "Account": 0.20,
    "Network": 0.18,
    "Hardware": 0.15,
    "Software": 0.17,
    "Security": 0.15,
    "Other": 0.15,
}

SPLIT_RATIOS: Dict[str, float] = {"train": 0.70, "validation": 0.15, "test": 0.15}

OPENERS = [
    "Hi support team,",
    "Hello,",
    "Dear helpdesk,",
    "Hi,",
    "To whom it may concern,",
    "Good morning,",
]

CLOSERS = [
    "Please help as soon as possible.",
    "Any help would be appreciated.",
    "Let me know what you need from me.",
    "This is affecting my work today.",
    "Thanks in advance for your help.",
    "I would like this resolved soon.",
]

# Small, class-neutral filler sentences. Mixed in at low probability so the
# corpus is not perfectly clean bag-of-words data (F1 should not be trivially
# 100%), without ever adding PII.
DISTRACTORS = [
    "By the way, thank you for the fast response last time.",
    "I will be out of office tomorrow afternoon.",
    "This has happened once before a few months ago.",
    "Our team is currently in the middle of a big deadline.",
    "I appreciate the team's help with previous tickets.",
    "This is the second time I am reporting a similar issue this quarter.",
]

# Deliberately overlapping vocabulary across classes: "password"/"login" in
# Account and Security, "update" in Software and Hardware, "account" in
# Account and Security, "connection"/"server" in Network and Software.
CLASS_BODIES: Dict[str, List[str]] = {
    "Account": [
        "I cannot log into my account even though I am sure the password is correct.",
        "My account got locked after too many failed login attempts.",
        "I forgot my username and need help recovering account access.",
        "The system says my account is disabled and I do not know why.",
        "I need to reset my password but the reset link never arrives.",
        "My profile information is missing after the last account update.",
        "I was logged out of every application and cannot sign back in.",
        "Two-factor authentication is not sending a code to my registered device.",
        "My account was merged with another user and now settings are wrong.",
        "The password reset email for my account never shows up in my inbox.",
    ],
    "Network": [
        "The office wifi keeps disconnecting every few minutes.",
        "I cannot connect to the VPN from home today.",
        "Internet speed on the third floor is extremely slow this morning.",
        "My laptop shows connected to wifi but no pages will load.",
        "The shared network drive is unreachable from my desktop.",
        "Video calls keep freezing because of a poor connection.",
        "I get a network unreachable error when opening internal tools.",
        "The ethernet port in the meeting room does not seem to work.",
        "Our office router seems to restart on its own several times a day.",
        "The remote desktop connection drops every time I try to log in.",
    ],
    "Hardware": [
        "My laptop screen stays black even after a full restart.",
        "The office printer on the second floor is jammed again.",
        "My keyboard has several keys that stopped responding.",
        "The monitor keeps flickering and sometimes turns off completely.",
        "My laptop battery drains fully within thirty minutes.",
        "The docking station is not charging my laptop anymore.",
        "My mouse cursor freezes randomly during the day.",
        "The webcam on my laptop is not detected by any application.",
        "The laptop fan is extremely loud and the case feels hot.",
        "A firmware update left my laptop stuck on the boot screen.",
    ],
    "Software": [
        "The reporting application crashes every time I try to export a file.",
        "My license for the design software appears to have expired.",
        "The latest update broke the spreadsheet macros I rely on.",
        "The installer fails halfway through with an unknown error code.",
        "The mobile app keeps freezing on the login screen.",
        "I cannot open any document since the last software update.",
        "The calendar sync stopped working after I updated the client.",
        "An error dialog appears every time I open the accounting tool.",
        "The internal dashboard shows a blank page after the server update.",
        "A recent patch removed a feature our whole team depends on.",
    ],
    "Security": [
        "I received a suspicious email asking me to confirm my password.",
        "My account shows a login attempt from a location I do not recognize.",
        "I think my laptop might be infected after clicking a strange link.",
        "A colleague reported that someone is impersonating our support team.",
        "I noticed unfamiliar software running in the background of my computer.",
        "The antivirus alert keeps flagging a file I did not download.",
        "I accidentally shared my password with what looked like an internal form.",
        "Our shared folder permissions changed without anyone approving it.",
        "I got a message claiming my account will be closed unless I verify it immediately.",
        "Someone asked me over chat to send my login details for a system check.",
    ],
    "Other": [
        "I would like to request a new desk chair for the office.",
        "Can someone update the meeting room booking calendar for next week.",
        "I have a general question about the new office badge policy.",
        "The coffee machine on our floor is out of order again.",
        "I would like feedback on the recent helpdesk satisfaction survey.",
        "Could you confirm the parking policy for visitors next month.",
        "I want to suggest an improvement for the internal ticketing process.",
        "Is there an update on the office relocation timeline.",
        "Can you tell me who to contact about a facilities request.",
        "I have a question about expense reimbursement for a work trip.",
    ],
}

# Harder, less keyword-obvious phrasing used for a portion of every split so
# a bag-of-words baseline does not reach a trivial 100% F1.
CLASS_PARAPHRASES: Dict[str, List[str]] = {
    "Account": [
        "Something is wrong with how I sign in and I keep getting rejected.",
        "None of my saved credentials seem to work anymore this week.",
    ],
    "Network": [
        "Everything online feels disconnected today, on and off all morning.",
        "Pages load, then stop loading, then load again without any pattern.",
    ],
    "Hardware": [
        "The physical machine on my desk is behaving strangely since yesterday.",
        "Something on my desk setup just stopped turning on properly.",
    ],
    "Software": [
        "The program I use daily behaves unexpectedly since the newest release.",
        "A tool I rely on stopped doing what it used to do correctly.",
    ],
    "Security": [
        "Something about a recent message felt off and I want a second opinion.",
        "I am worried something unusual happened with my access this week.",
    ],
    "Other": [
        "I have a request that does not really fit any technical category.",
        "This is more of a general office matter than a computer problem.",
    ],
}

HARD_CASE_KIND_NOTE = {
    "typo": "typo robustness",
    "paraphrase": "paraphrase robustness",
    "irrelevant": "irrelevant-text robustness",
    "ambiguous": "ambiguous-keyword robustness",
}


def _weighted_counts(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    """Deterministic class counts that sum exactly to ``total``."""
    raw = {cls: total * weight for cls, weight in weights.items()}
    counts = {cls: int(value) for cls, value in raw.items()}
    remainder = total - sum(counts.values())
    # Distribute the remainder to the classes with the largest fractional part,
    # breaking ties by class name so the result is deterministic.
    fractional_order = sorted(
        CLASSES, key=lambda cls: (-(raw[cls] - counts[cls]), cls)
    )
    for cls in fractional_order[:remainder]:
        counts[cls] += 1
    return counts


def _make_ticket_text(rng: random.Random, body: str, allow_distractor: bool = True) -> str:
    parts = [rng.choice(OPENERS), body]
    if allow_distractor and rng.random() < 0.18:
        parts.append(rng.choice(DISTRACTORS))
    parts.append(rng.choice(CLOSERS))
    return " ".join(parts)


def _typo(rng: random.Random, text: str) -> str:
    """Deterministic, mild character-level corruption (swap or drop a
    letter in 1-2 words) — not a missing-diacritics transform."""
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) > 4]
    if not candidates:
        return text
    n_typos = 1 if len(candidates) < 3 else 2
    targets = rng.sample(candidates, n_typos)
    for idx in targets:
        word = words[idx]
        letters = list(word)
        pos = rng.randrange(1, len(letters) - 1)
        mode = rng.choice(["swap", "drop"])
        if mode == "swap" and pos + 1 < len(letters):
            letters[pos], letters[pos + 1] = letters[pos + 1], letters[pos]
        else:
            del letters[pos]
        words[idx] = "".join(letters)
    return " ".join(words)


def split_dataset(rng: random.Random, total: int) -> Dict[str, List[Dict[str, str]]]:
    """Generate rows per class, then split each class by SPLIT_RATIOS so
    every split keeps a similar (not extremely skewed) label distribution."""
    counts = _weighted_counts(total, CLASS_WEIGHTS)
    splits: Dict[str, List[Dict[str, str]]] = {name: [] for name in SPLIT_RATIOS}

    for cls in CLASSES:
        bodies = CLASS_BODIES[cls]
        paraphrases = CLASS_PARAPHRASES[cls]
        n = counts[cls]
        n_paraphrase = max(1, round(n * 0.08))
        class_rows = []
        for i in range(n):
            body = rng.choice(paraphrases) if i < n_paraphrase else rng.choice(bodies)
            class_rows.append({"text": _make_ticket_text(rng, body), "label": cls})
        rng.shuffle(class_rows)

        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["validation"])
        # test gets the remainder so counts always sum to n.
        train_rows = class_rows[:n_train]
        val_rows = class_rows[n_train : n_train + n_val]
        test_rows = class_rows[n_train + n_val :]
        splits["train"].extend(train_rows)
        splits["validation"].extend(val_rows)
        splits["test"].extend(test_rows)

    for name in splits:
        rng.shuffle(splits[name])
    return splits


def generate_hard_cases(rng: random.Random) -> List[Dict[str, str]]:
    """Build the robustness set: for every class, one typo case, one
    paraphrase case, one irrelevant-text-padded case, and one
    ambiguous-keyword case (English robustness — no diacritics stripping)."""
    rows: List[Dict[str, str]] = []
    class_pairs = list(zip(CLASSES, CLASSES[1:] + CLASSES[:1]))  # for ambiguous mixing

    for cls in CLASSES:
        base_body = CLASS_BODIES[cls][0]
        typo_text = _make_ticket_text(rng, _typo(rng, base_body), allow_distractor=False)
        rows.append({"text": typo_text, "label": cls, "_kind": "typo"})

        paraphrase_text = _make_ticket_text(rng, rng.choice(CLASS_PARAPHRASES[cls]), allow_distractor=False)
        rows.append({"text": paraphrase_text, "label": cls, "_kind": "paraphrase"})

        padded = " ".join(
            [rng.choice(DISTRACTORS), rng.choice(DISTRACTORS), base_body, rng.choice(DISTRACTORS)]
        )
        rows.append({"text": padded, "label": cls, "_kind": "irrelevant"})

    for cls, other_cls in class_pairs:
        ambiguous_text = _make_ticket_text(
            rng,
            f"{CLASS_BODIES[cls][1]} It might also be related to {CLASS_BODIES[other_cls][1].lower()}",
            allow_distractor=False,
        )
        rows.append({"text": ambiguous_text, "label": cls, "_kind": "ambiguous"})

    return rows


def _write_csv(path: Path, rows: List[Dict[str, str]], id_prefix: str, start_index: int = 1) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["ticket_id", "text", "label"])
        idx = start_index
        for row in rows:
            ticket_id = f"{id_prefix}-{idx:06d}"
            writer.writerow([ticket_id, row["text"], row["label"]])
            idx += 1
    return idx


def _write_sha256sums(out_dir: Path, filenames: List[str]) -> None:
    lines = []
    for name in sorted(filenames):
        digest = hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}\n")
    with open(out_dir / "SHA256SUMS", "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--total",
        type=int,
        default=3000,
        help="Total tickets across train+validation+test, 1500-5000 (default: 3000)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for the CSV files (default: project data/)",
    )
    args = parser.parse_args()

    if not (1500 <= args.total <= 5000):
        raise SystemExit(f"--total must be between 1500 and 5000, got {args.total}")

    rng = random.Random(args.seed)
    splits = split_dataset(rng, args.total)
    hard_case_rows = generate_hard_cases(rng)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(out_dir / "train.csv", splits["train"], id_prefix="TRN")
    _write_csv(out_dir / "validation.csv", splits["validation"], id_prefix="VAL")
    _write_csv(out_dir / "test.csv", splits["test"], id_prefix="TST")
    _write_csv(
        out_dir / "hard_cases.csv",
        [{"text": r["text"], "label": r["label"]} for r in hard_case_rows],
        id_prefix="HARD",
    )

    _write_sha256sums(out_dir, ["train.csv", "validation.csv", "test.csv", "hard_cases.csv"])

    print(f"Wrote {sum(len(v) for v in splits.values())} tickets + {len(hard_case_rows)} hard cases to {out_dir}")
    for name, rows in splits.items():
        counts = {cls: sum(1 for r in rows if r["label"] == cls) for cls in CLASSES}
        print(f"  {name}: {len(rows)} rows -> {counts}")


if __name__ == "__main__":
    main()
