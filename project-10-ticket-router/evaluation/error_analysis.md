# Qualitative error analysis — Project 10 (Midterm)

Companion to `results.md` (all figures reproducible via
`evaluation/run_experiments.py`).

## 0. Where the errors are — and where they are not

Trained on the full training split, **all three pipelines make zero
errors on validation (448) and test (452)** — accuracy and macro-F1 are
1.0000 everywhere, and the confusion matrices are perfectly diagonal.
Because `data/label_description.md` explicitly warns that a trivial
100% should be treated as a leakage symptom, we investigated before
trusting it:

- The pipelines fit **only** on `train.csv` (vocabulary, IDF weights
  and classifier parameters never see validation/test), so there is no
  fitting leakage.
- The cause is the split itself: the corpus is template-generated and
  split randomly. **40.3% of test texts appear verbatim in train.csv**,
  and a test ticket's token set has a mean best-Jaccard similarity of
  **0.893** to a same-label train ticket. Validation/test therefore
  mostly measure template memorisation.
- Real errors survive in exactly two places, and those are what this
  analysis covers: **(A)** the `ambiguous` group of `hard_cases.csv`,
  and **(B)** a data-efficiency probe — the selected pipeline
  (`tfidf_svm`) retrained on a random 10% of train (210 tickets),
  which scores 0.9381 on test with 28 errors.

## A. Ambiguous hard cases (keyword conflicts)

All 24 hard cases of the *typo*, *paraphrase* and *irrelevant-filler*
categories are classified correctly by every pipeline, most with
confidence margins above 0.84. **Every hard-case error is in the
*ambiguous* group (HARD-000019…24)**, where each ticket states a
primary issue and then adds "It might also be related to …" plus a
second class's issue. A bag-of-words model cannot see which issue is
primary — it can only weigh total keyword mass.

1. **HARD-000021 — true Hardware, predicted Software (all 3 pipelines).**
   "The office printer on the second floor is jammed again. It might
   also be related to my license for the design software appears to
   have expired." The secondary sentence is a complete, canonical
   Software template (*license / software / expired* — three strong
   Software features) and outweighs *printer / jammed*. The SVM's
   top-2 margin is only 0.034: a coin flip that landed wrong.
2. **HARD-000020 — true Network, predicted Hardware (bow_nb, tfidf_svm).**
   "I cannot connect to the VPN from home today. It might also be
   related to the office printer on the second floor is jammed
   again." *printer* and *jammed* are rare, high-IDF Hardware words;
   the genuine Network evidence (*connect / VPN*) is shorter, so the
   secondary issue wins on keyword mass. Logistic Regression (with
   character features) gets it right, but only at 0.557 vs 0.419.
3. **HARD-000022 — true Software, predicted Security (bow_nb, tfidf_logreg).**
   "My license for the design software appears to have expired. It
   might also be related to my account shows a login attempt from a
   location I do not recognize." The secondary clause is the single
   most canonical Security template in the corpus (*login attempt /
   location / do not recognize*). Notably the SVM classifies it
   correctly — with a margin of **0.015**, i.e. it is right by luck,
   not by understanding.
4. **HARD-000024 — true Other, predicted Account (bow_nb, tfidf_logreg).**
   "Can someone update the meeting room booking calendar for next
   week. It might also be related to my account got locked after too
   many failed login attempts." Same failure shape: a non-technical
   request (Other) loses to an embedded canonical Account template.
5. **Near-misses HARD-000019 (Account) and HARD-000023 (Security)** are
   classified correctly by all pipelines, but the SVM margins are
   0.079 and 0.034. Together with cases 1–4 this shows the whole
   ambiguous group sits at the decision boundary (margins 0.015–0.079
   for the SVM). **Design consequence for the Final app:** these are
   exactly the tickets the required "top-two predictions are close"
   alert exists for — the honest system behaviour is to show both
   candidate queues and let a human confirm, not to pretend certainty.

**Pattern A:** ambiguity errors are *structural*, not statistical.
More training data does not fix them (they occur at full data), because
the missing information is discourse structure — which issue is
primary — and BoW/TF-IDF representations discard word order beyond
short n-grams and all discourse markers.

## B. Data-efficiency probe errors (paraphrases starved of data)

With only 210 training tickets the selected pipeline makes 28 test
errors. The instructive ones:

6. **TST-000138 / TST-000142 — true Software, predicted Account.**
   "The program I use daily behaves unexpectedly since the newest
   release." A paraphrase template with *no* strong Software keyword
   (no *crash*, *install*, *app name*). With little data the weak words
   (*program, release*) do not accumulate enough weight, and the model
   falls back to a wrong prior. At full data the same texts are
   classified correctly — data volume, not model class, is the fix.
7. **TST-000051 / TST-000143 — true Hardware, predicted Network.**
   "My laptop battery drains fully within thirty minutes." In the
   210-ticket subsample the *battery drains* template is rare, so
   *battery* never earns a decisive Hardware weight and *laptop* alone
   cannot carry the class.
8. **TST-000020 — true Software, predicted Other; TST-000148 — true
   Software, predicted Security.** "The internal dashboard shows a
   blank page after the server update." *server* belongs to the
   Network/Software shared vocabulary described in the taxonomy, and
   *blank page* is unseen in the subsample — the ticket is pulled
   toward whichever neighbouring class the few observed words favour.
9. **TST-000035 / TST-000163 — true Software, predicted Other /
   Security.** "A recent patch removed a feature our whole team
   depends on." Another paraphrase: *patch* is the only Software cue,
   and it is too rare in the subsample to win.

**Pattern B:** probe errors concentrate on (i) **paraphrased tickets**
whose class keyword was deliberately removed (the ~8% paraphrase slice
of the corpus) and (ii) **shared-vocabulary words** (*server*,
*update*, *account*) — exactly the two difficulties the taxonomy in
`label_description.md` is designed to probe. Unlike Pattern A, these
errors vanish as training data grows (28 errors at 210 tickets → 0 at
2,100), which is the expected learning-curve behaviour and confirms
the full-data result is memorisation of a finite template inventory
rather than a bug.

## C. Robustness summary (from `results.md` §2)

| Pipeline | Test acc | Hard-case acc | Degradation | Errors by category |
|---|---|---|---|---|
| bow_nb | 1.0000 | 0.8333 | −16.7 pts | ambiguous only (4/6 wrong) |
| tfidf_logreg | 1.0000 | 0.8750 | −12.5 pts | ambiguous only (3/6 wrong) |
| **tfidf_svm (selected)** | 1.0000 | **0.9167** | **−8.3 pts** | ambiguous only (2/6 wrong) |

Typographical noise costs nothing measurable here: every typo case is
still classified correctly because a corrupted word (*acconut*,
*passowrd*) never appears alone — the rest of the ticket keeps clean,
high-weight keywords, and character n-grams add a second safety net.
The measured "robustness gap" of this system is therefore **semantic
ambiguity, not spelling noise** — the finding the midterm demo should
lead with.
