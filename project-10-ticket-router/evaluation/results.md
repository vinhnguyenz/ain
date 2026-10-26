# Midterm experiment results — Project 10 (Ticket Router)

All numbers produced by `evaluation/run_experiments.py` (train n=2100, validation n=448, test n=452, hard_cases n=24). Model selection used the **validation** split; test metrics are reported once, for the selected pipeline.

## 1. Pipeline comparison (macro-averaged)

| Variant | Split | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| bow_nb | validation | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| bow_nb | test | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tfidf_svm | validation | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tfidf_svm | test | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tfidf_logreg | validation | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tfidf_logreg | test | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Selected pipeline (validation macro-F1, tie-break = hard-case robustness): **tfidf_svm**.

## 1b. Why are validation/test scores perfect? (split-overlap check)

`label_description.md` warns that 100% F1 should trigger a leakage check. Our pipelines fit **only** on the training split (vectorizer vocabulary and classifier are learned from train.csv alone), so there is no fitting leakage. The perfect scores come from the split itself: the corpus is template-generated and split randomly, so near-identical template siblings land in both train and test.

- Exact-duplicate rate vs train: validation 42.4%, test 40.3% of texts appear verbatim in train.csv.
- Mean best Jaccard similarity between a test ticket's token set and a same-label train ticket: 0.893 (sample of 60).

Conclusion: validation/test measure template memorisation more than generalisation here; `hard_cases.csv` and the ambiguous cases are the discriminative evaluation, and the data-efficiency probe in section 4 shows how the selected pipeline behaves when memorisation is limited.

## 2. Robustness on hard_cases.csv

| Variant | Hard accuracy | Test accuracy | Degradation |
|---|---|---|---|
| bow_nb | 0.8333 | 1.0000 | +0.1667 |
| tfidf_svm | 0.9167 | 1.0000 | +0.0833 |
| tfidf_logreg | 0.8750 | 1.0000 | +0.1250 |

Per-category accuracy (hard cases):

| Variant | Category | n | Accuracy |
|---|---|---|---|
| bow_nb | typo | 6 | 1.0000 |
| bow_nb | paraphrase | 6 | 1.0000 |
| bow_nb | irrelevant | 6 | 1.0000 |
| bow_nb | ambiguous | 6 | 0.3333 |
| tfidf_logreg | typo | 6 | 1.0000 |
| tfidf_logreg | paraphrase | 6 | 1.0000 |
| tfidf_logreg | irrelevant | 6 | 1.0000 |
| tfidf_logreg | ambiguous | 6 | 0.5000 |
| tfidf_svm | typo | 6 | 1.0000 |
| tfidf_svm | paraphrase | 6 | 1.0000 |
| tfidf_svm | irrelevant | 6 | 1.0000 |
| tfidf_svm | ambiguous | 6 | 0.6667 |

Hard-case detail for the selected pipeline:

| ID | Category | True | Predicted | Correct | Text |
|---|---|---|---|---|---|
| HARD-000001 | typo | Account | Account | yes | Hi support team, I cannot log into my acconut even though I am sure the passowrd is correct. Let me know what you need from me. |
| HARD-000002 | paraphrase | Account | Account | yes | Hi support team, None of my saved credentials seem to work anymore this week. Thanks in advance for your help. |
| HARD-000003 | irrelevant | Account | Account | yes | This is the second time I am reporting a similar issue this quarter. This is the second time I am reporting a similar issue this q |
| HARD-000004 | typo | Network | Network | yes | Hello, The offcie wifi keeps disconnecting every few minuets. This is affecting my work today. |
| HARD-000005 | paraphrase | Network | Network | yes | Hello, Pages load, then stop loading, then load again without any pattern. Thanks in advance for your help. |
| HARD-000006 | irrelevant | Network | Network | yes | This is the second time I am reporting a similar issue this quarter. I will be out of office tomorrow afternoon. The office wifi k |
| HARD-000007 | typo | Hardware | Hardware | yes | Hi, My laptop screen stays blakc even after a full rstart. Please help as soon as possible. |
| HARD-000008 | paraphrase | Hardware | Hardware | yes | Hello, The physical machine on my desk is behaving strangely since yesterday. I would like this resolved soon. |
| HARD-000009 | irrelevant | Hardware | Hardware | yes | This is the second time I am reporting a similar issue this quarter. This has happened once before a few months ago. My laptop scr |
| HARD-000010 | typo | Software | Software | yes | Hi, The reporting application crasehs every time I try to export a fil. Any help would be appreciated. |
| HARD-000011 | paraphrase | Software | Software | yes | Dear helpdesk, The program I use daily behaves unexpectedly since the newest release. Please help as soon as possible. |
| HARD-000012 | irrelevant | Software | Software | yes | This has happened once before a few months ago. This has happened once before a few months ago. The reporting application crashes  |
| HARD-000013 | typo | Security | Security | yes | Hi support team, I received a suspicious email asking me to cnofirm my passwodr. Please help as soon as possible. |
| HARD-000014 | paraphrase | Security | Security | yes | Hi support team, Something about a recent message felt off and I want a second opinion. Let me know what you need from me. |
| HARD-000015 | irrelevant | Security | Security | yes | I will be out of office tomorrow afternoon. By the way, thank you for the fast response last time. I received a suspicious email a |
| HARD-000016 | typo | Other | Other | yes | Hi, I woudl like to request a new desk char for the office. I would like this resolved soon. |
| HARD-000017 | paraphrase | Other | Other | yes | Hi support team, This is more of a general office matter than a computer problem. Let me know what you need from me. |
| HARD-000018 | irrelevant | Other | Other | yes | By the way, thank you for the fast response last time. This has happened once before a few months ago. I would like to request a n |
| HARD-000019 | ambiguous | Account | Account | yes | Hi, My account got locked after too many failed login attempts. It might also be related to i cannot connect to the vpn from home  |
| HARD-000020 | ambiguous | Network | Hardware | NO | To whom it may concern, I cannot connect to the VPN from home today. It might also be related to the office printer on the second  |
| HARD-000021 | ambiguous | Hardware | Software | NO | Good morning, The office printer on the second floor is jammed again. It might also be related to my license for the design softwa |
| HARD-000022 | ambiguous | Software | Software | yes | To whom it may concern, My license for the design software appears to have expired. It might also be related to my account shows a |
| HARD-000023 | ambiguous | Security | Security | yes | Hi support team, My account shows a login attempt from a location I do not recognize. It might also be related to can someone upda |
| HARD-000024 | ambiguous | Other | Other | yes | Hello, Can someone update the meeting room booking calendar for next week. It might also be related to my account got locked after |

## 3. Test-set misclassifications of the selected pipeline

Total errors: 0 / 452.

| ID | True | Predicted | Text |
|---|---|---|---|

Confusion matrices are saved as `cm_<variant>_<split>.png` next to this file. With zero errors on validation/test they are perfectly diagonal; the informative matrix is the hard-case one implied by section 2.

## 4. Data-efficiency probe (10% of train) — error-analysis material

Selected pipeline retrained on a random 10% subsample of train (n=210, seed 7): test accuracy 0.9381 with 28 errors (vs 0 errors with the full training set). These misclassifications are analysed qualitatively in `error_analysis.md`.

| ID | True | Predicted | Text |
|---|---|---|---|
| TST-000020 | Software | Other | Hello, The internal dashboard shows a blank page after the server update. Our team is currently in the middle of a big deadline. Any help would be appreciated. |
| TST-000035 | Software | Other | Hi support team, A recent patch removed a feature our whole team depends on. I appreciate the team's help with previous tickets. This is affecting my work today |
| TST-000051 | Hardware | Network | Hello, My laptop battery drains fully within thirty minutes. Let me know what you need from me. |
| TST-000138 | Software | Account | Dear helpdesk, The program I use daily behaves unexpectedly since the newest release. Let me know what you need from me. |
| TST-000142 | Software | Account | Dear helpdesk, The program I use daily behaves unexpectedly since the newest release. Any help would be appreciated. |
| TST-000143 | Hardware | Network | Hi, My laptop battery drains fully within thirty minutes. This is the second time I am reporting a similar issue this quarter. Let me know what you need from me |
| TST-000148 | Software | Security | Good morning, The internal dashboard shows a blank page after the server update. Thanks in advance for your help. |
| TST-000163 | Software | Security | Hello, A recent patch removed a feature our whole team depends on. Please help as soon as possible. |
| TST-000164 | Software | Security | Hello, A recent patch removed a feature our whole team depends on. This is the second time I am reporting a similar issue this quarter. This is affecting my wor |
| TST-000182 | Other | Security | Hello, I want to suggest an improvement for the internal ticketing process. Thanks in advance for your help. |
| TST-000187 | Hardware | Network | Hello, My mouse cursor freezes randomly during the day. Any help would be appreciated. |
| TST-000200 | Software | Security | Good morning, A recent patch removed a feature our whole team depends on. Please help as soon as possible. |
| TST-000227 | Hardware | Network | Hi support team, My mouse cursor freezes randomly during the day. This is affecting my work today. |
| TST-000238 | Hardware | Network | Hi, My laptop battery drains fully within thirty minutes. Our team is currently in the middle of a big deadline. Any help would be appreciated. |
| TST-000242 | Software | Account | Hi, The program I use daily behaves unexpectedly since the newest release. I would like this resolved soon. |
| TST-000262 | Software | Security | Hi, A recent patch removed a feature our whole team depends on. Our team is currently in the middle of a big deadline. Let me know what you need from me. |
| TST-000269 | Software | Security | Good morning, A recent patch removed a feature our whole team depends on. Please help as soon as possible. |
| TST-000276 | Other | Security | Hello, I want to suggest an improvement for the internal ticketing process. Thanks in advance for your help. |
| TST-000291 | Software | Other | To whom it may concern, The internal dashboard shows a blank page after the server update. Please help as soon as possible. |
| TST-000324 | Software | Other | To whom it may concern, The internal dashboard shows a blank page after the server update. Thanks in advance for your help. |
| TST-000341 | Software | Other | To whom it may concern, The internal dashboard shows a blank page after the server update. Any help would be appreciated. |
| TST-000352 | Hardware | Software | To whom it may concern, My laptop battery drains fully within thirty minutes. This has happened once before a few months ago. Thanks in advance for your help. |
| TST-000360 | Hardware | Network | To whom it may concern, My laptop battery drains fully within thirty minutes. Any help would be appreciated. |
| TST-000371 | Software | Security | Hello, A recent patch removed a feature our whole team depends on. Let me know what you need from me. |
| TST-000391 | Software | Other | Good morning, A recent patch removed a feature our whole team depends on. This is affecting my work today. |
| TST-000393 | Network | Security | Dear helpdesk, Everything online feels disconnected today, on and off all morning. I will be out of office tomorrow afternoon. Thanks in advance for your help. |
| TST-000411 | Software | Network | Good morning, The program I use daily behaves unexpectedly since the newest release. Let me know what you need from me. |
| TST-000413 | Software | Security | Good morning, A tool I rely on stopped doing what it used to do correctly. I will be out of office tomorrow afternoon. This is affecting my work today. |

### Per-class report (selected pipeline, test split)

```
              precision    recall  f1-score   support

     Account       1.00      1.00      1.00        90
     Network       1.00      1.00      1.00        81
    Hardware       1.00      1.00      1.00        68
    Software       1.00      1.00      1.00        77
    Security       1.00      1.00      1.00        68
       Other       1.00      1.00      1.00        68

    accuracy                           1.00       452
   macro avg       1.00      1.00      1.00       452
weighted avg       1.00      1.00      1.00       452
```
