# Project 10 — IT Helpdesk Ticket Router

## 1. Problem Description

Supervised text classification in Natural Language Processing (NLP) combines vector space representations with statistical Machine Learning (ML) algorithms to categorize unstructured text into predefined discrete classes. Feature pipelines such as Bag-of-Words (BoW) or Term Frequency-Inverse Document Frequency (TF-IDF) map raw token sequences into structured feature vectors, enabling discriminative or generative classifiers to learn decision boundaries across high-dimensional semantic spaces.

This project automates the routing of English technical support tickets submitted to a campus IT helpdesk. Each free-text incident report must be triaged into exactly one of six operational support queues:
$$\mathcal{Y} = \{\text{Account}, \text{Network}, \text{Hardware}, \text{Software}, \text{Security}, \text{Other}\}$$

Technical vocabulary frequently overlaps across categories (e.g., `password` and `login` appearing in both `Account` and `Security`; `cable` in both `Network` and `Hardware`), so classifiers must learn nuanced feature weights and robust probability thresholds. Students will implement feature extraction pipelines and supervised classifiers, conduct formal error analysis, and stress-test models against noisy, paraphrased, and ambiguous inputs.

---

## 2. Provided Materials & Starter Resources

The project package provides labeled synthetic ticket datasets, tokenization utilities, and an interactive Streamlit application.

### File Structure
```text
project-10-ticket-router/
├── data/
│   ├── label_description.md      # Taxonomy guide, shared vocabulary analysis, and partition statistics
│   ├── train.csv                 # Supervised training split (ticket_id, text, label)
│   ├── validation.csv            # Tuning and hyperparameter selection split
│   ├── test.csv                  # Unseen evaluation holdout split
│   ├── hard_cases.csv            # 24 adversarial fixtures (typos, paraphrasing, keyword conflicts)
│   └── SHA256SUMS                # Cryptographic checksums ensuring data reproducibility
├── scripts/
│   └── generate_tickets.py       # Reproducible synthetic ticket generator (default seed 42)
├── starter/
│   ├── tokenizer.py              # Text preprocessing utilities (tokenization, vocabulary, vectorization)
│   ├── student_core.py           # Algorithmic stubs: train_model and predict
│   └── app.py                    # Streamlit ticket routing and evaluation dashboard
├── tests/
│   └── test_project_10_sanity.py # Unit tests validating split shapes, labels, and stub contracts
└── requirements.txt              # Pinned Python package dependencies (including scikit-learn, streamlit)
```

### Starter Infrastructure vs. Student Implementation
`starter/tokenizer.py` provides optional baseline tokenization and count-vectorization helpers. `starter/app.py` renders an interactive support-ticketing interface. Students implement the ML pipeline in `starter/student_core.py`:

- **`train_model(train_path, validation_path)`** — builds and fits a complete text classification pipeline (feature extraction + classifier) on the provided CSV splits.
- **`predict(model, text)`** — returns the predicted label and a confidence score for a single free-text ticket.

Students may use `scikit-learn` algorithms (Multinomial Naive Bayes, Linear SVM, Logistic Regression). Shipping pre-fitted binary weights (`.pkl`, `.joblib`) or querying external LLM APIs is strictly prohibited.

### Installation and Execution
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run starter/app.py
```

---

## 3. Requirements & Deliverables

### 3.1 Midterm Milestone — AI Core
- **Algorithmic Implementation**: Build an end-to-end text classification pipeline in `starter/student_core.py`. Implement `train_model` performing feature engineering (tokenization, stopword filtering, n-grams, BoW or TF-IDF) and supervised classifier training on `data/train.csv`. Implement `predict` to return a label and confidence for any input string. Compare at least two distinct pipeline combinations (e.g., BoW + Multinomial Naive Bayes vs. TF-IDF + Linear SVM).
- **Mandatory Controlled Experiment**: Evaluate both pipelines on `data/validation.csv` and `data/test.csv`. Report macro-averaged Accuracy, Precision, Recall, and F1-score alongside a multi-class Confusion Matrix. Perform qualitative error analysis on at least five misclassified tickets. Evaluate robustness on `data/hard_cases.csv`, quantifying accuracy degradation under typographical noise and cross-domain keyword confusion.
- **Deliverables**: Implemented `student_core.py`, training/evaluation scripts or notebooks, and `presentation.pdf`.
- **Grading**: Feature pipeline and classification correctness (15 pts); experimental rigor, confusion analysis, and robustness benchmarking (10 pts); oral defense with live ticket classification demo (15 pts). **Total: 40 pts.**

### 3.2 Final Milestone — AI Product
- **Interactive Application**: Package the classifier into a Streamlit web application. Required features: free-text ticket entry with real-time queue prediction; predicted class confidence bars; secondary alert when top-two predictions are within a small margin; test-split ticket browser; embedded evaluation dashboard with validation metrics and confusion matrix.
- **Stress & Edge Scenarios**: Demonstrate live routing across four benchmark archetypes: clear unambiguous requests, ambiguous multi-class text, typo-corrupted tickets, and degenerate inputs (empty or single-character).
- **Deliverables**: Functional Streamlit application, clean GitHub repository, and `report.pdf` (IEEE format).
- **Grading**: UI ergonomics, triage workflow clarity, and confidence transparency (25 pts); report mathematical rigor and lexical analysis (15 pts); oral defense against adversarial edge cases (10 pts). **Total: 50 pts.**

---

## 4. References

- [1] A. McCallum and K. Nigam, "A comparison of event models for naive Bayes text classification," in *Proc. AAAI-98 Workshop on Learning for Text Categorization*, Madison, WI, USA, 1998, pp. 41–48.
- [2] T. Joachims, "Text categorization with support vector machines: Learning with many relevant features," in *Proc. 10th European Conf. Machine Learning (ECML)*, Chemnitz, Germany, 1998, pp. 137–142.
- [3] S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. Hoboken, NJ, USA: Pearson, 2020, pp. 828–864.
