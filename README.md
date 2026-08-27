# SpendSense

Sorts bank transactions into spending categories using a trained text
classifier, then shows where the money went.

---

## The problem

A bank statement is 200 lines of merchant names. Nobody reads it, so
nobody actually knows where their money goes until the month is over.

Sorting those lines by hand is the obvious fix and the reason nobody does
it. This does it automatically.

---

## What it does

1. Upload a CSV statement, or type a single transaction
2. A trained model reads each description and assigns a category
3. Everything is stored, totalled, and shown by category
4. Transactions the model was unsure about are flagged so you can correct them

Nine categories: food, travel, shopping, utilities, entertainment,
healthcare, education, investment, emi.

---

## Running it

```bash
pip install pandas scikit-learn joblib fastapi uvicorn python-multipart

python augment_data.py     # builds extra training data
python train_model.py      # trains and saves model.pkl
uvicorn app:app --reload   # starts the server
```

Open http://localhost:8000 and upload `sample_statement.csv` to see it work.

---

## The model

| | |
|---|---|
| Input | transaction description text |
| Output | one of 9 categories |
| Features | TF-IDF over words and word pairs |
| Classifier | Logistic Regression |
| Training rows | 6,182 |

TF-IDF converts text into numbers by weighting each word by how
distinctive it is. A word like "INR" appears in every row and carries no
information, so it gets weighted down. "Swiggy" appears in one category
only, so it gets weighted up. Word pairs are included as well, so
"home loan" is treated as a signal in its own right rather than as
"home" plus "loan" separately.

---

## Accuracy, and why one of these numbers is worthless

| Test set | Accuracy |
|---|---|
| Kaggle's own test split | 100.0% |
| 26 merchants seen in neither training file | **76.9%** |

The 100% is meaningless. The Kaggle dataset contains only **45 distinct
phrase templates** across all 6,000 rows — the rest of each row is a
random amount and a random transaction ID. The test split reuses the same
45 templates. So a model can score 100% by memorising a lookup table
without learning anything about language at all.

The 76.9% is measured against brands that appear nowhere in training:
Faasos, Vistara, Netmeds, PhysicsWallah, Kuvera and others. That is the
number that says whether the model actually works.

**Adding merchant variety moved this from 69.0% to 76.9%.** Training on
the Kaggle data alone gave 69.0%. `augment_data.py` adds roughly 200 more
real Indian merchants across the same nine categories, which gave the
model enough variety to generalise rather than memorise.

---

## Where it still fails

The six remaining errors are all the same failure mode: a brand name the
model has never encountered, with no other informative words in the
description. "Kuvera portfolio invest" gets called food because none of
those tokens appeared in training and the classifier falls back to the
category with the strongest prior.

Three things would help, in rough order of value:

- **More merchant coverage.** The failures are unseen brands, so the
  direct fix is more brands. This scales badly by hand — a merchant
  directory or a public MCC (merchant category code) mapping would do it
  properly.
- **Character n-grams alongside word n-grams.** These pick up substrings,
  so "PharmEasy" and "pharmacy" share signal even though they are
  different words.
- **A confidence threshold with human correction.** The app already
  records a confidence score per prediction and exposes an endpoint for
  low-confidence rows. Routing those to the user and feeding the
  corrections back into training is how a real system improves over time.

---

## Stack

- **Model** — Python, scikit-learn
- **Backend** — FastAPI
- **Database** — SQLite
- **Frontend** — HTML, CSS, vanilla JavaScript

---

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/predict` | Categorise one transaction without saving |
| POST | `/api/transactions` | Categorise and save one transaction |
| POST | `/api/upload` | Upload a CSV, categorise and save every row |
| GET | `/api/transactions` | List saved transactions |
| GET | `/api/summary` | Totals per category |
| GET | `/api/low-confidence` | Transactions the model was unsure about |
| DELETE | `/api/transactions` | Clear everything |

---

## Files

```
augment_data.py    builds extra training rows from a merchant list
train_model.py     trains the classifier, reports both accuracy figures
real_test.py       standalone honest evaluation
app.py             FastAPI backend
index.html         single-page frontend
sample_statement.csv   20 rows to test the upload with
```

---

## Data

Base dataset: Financial Transaction Description Dataset (Kaggle,
bhavyasingh25) — 5,000 training and 1,000 test rows, synthetic.
Augmented with ~200 additional Indian merchant names written for this
project.

Both sources are synthetic. Real labelled bank transaction data is not
publicly available for privacy reasons, which is why every dataset in
this space is generated.
