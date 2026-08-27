"""
train_model.py - trains the transaction categoriser.

Reads two sources of labelled examples, learns the pattern, checks itself
on transactions it has never seen, and saves the result to model.pkl.

Run:  python train_model.py
"""

import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# ---- 1. Load the training data ---------------------------------------
kaggle = pd.read_csv("train_transactions.csv")

# The Kaggle set only contains 45 distinct phrases, so a model trained on
# it alone just memorises them. augmented_train.csv adds ~200 more real
# merchants so the model has enough variety to generalise.
if os.path.exists("augmented_train.csv"):
    extra = pd.read_csv("augmented_train.csv")
    train = pd.concat([kaggle, extra], ignore_index=True)
    print(f"Kaggle rows: {len(kaggle)}  +  augmented rows: {len(extra)}")
else:
    train = kaggle
    print("augmented_train.csv not found - run augment_data.py first for better results")

X_train = train["transaction_text"]    # the input
y_train = train["category"]            # the answer we want

print(f"Training on {len(train)} transactions")
print(f"Categories: {sorted(y_train.unique())}\n")


# ---- 2. Build the model ----------------------------------------------
# TfidfVectorizer turns words into numbers, because maths needs numbers.
#   ngram_range=(1,2) means it looks at single words AND word pairs,
#   so "home loan" is treated as its own signal, not just "home" + "loan".
# LogisticRegression then learns which words point to which category.
model = Pipeline([
    ("vectorizer", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000)),
])


# ---- 3. Train --------------------------------------------------------
model.fit(X_train, y_train)
print("Training done.\n")


# ---- 4. Check it on the Kaggle test set ------------------------------
test = pd.read_csv("test_transactions.csv")
preds = model.predict(test["transaction_text"])
kaggle_acc = accuracy_score(test["category"], preds)

print(f"Kaggle test set accuracy: {kaggle_acc * 100:.1f}%")
print("  (this number is inflated - the test set reuses the same 45")
print("   phrases as the training set, so the model can just memorise)\n")


# ---- 5. The honest check: merchants it has never seen ----------------
holdout = [
    ("Faasos wrap order INR 320 TXNa1", "food"),
    ("Behrouz biryani INR 640 TXNb2", "food"),
    ("Country Delight milk INR 180 TXNc3", "food"),
    ("Chaayos outlet INR 250 TXNd4", "food"),
    ("BluSmart electric cab INR 410 TXNe5", "travel"),
    ("Vistara airlines ticket INR 8900 TXNf6", "travel"),
    ("Cleartrip flight INR 5600 TXNg7", "travel"),
    ("Shell fuel station INR 3000 TXNh8", "travel"),
    ("Westside apparel INR 2100 TXNi9", "shopping"),
    ("FirstCry kids store INR 1450 TXNj1", "shopping"),
    ("Titan watch purchase INR 7800 TXNk2", "shopping"),
    ("Adani Electricity bill INR 2900 TXNl3", "utilities"),
    ("Hathway broadband INR 850 TXNm4", "utilities"),
    ("SonyLIV subscription INR 299 TXNn5", "entertainment"),
    ("Zee5 plan renewal INR 499 TXNo6", "entertainment"),
    ("Cinepolis tickets INR 900 TXNp7", "entertainment"),
    ("Netmeds order INR 720 TXNq8", "healthcare"),
    ("Fortis hospital charges INR 15600 TXNr9", "healthcare"),
    ("Thyrocare lab test INR 1100 TXNs1", "healthcare"),
    ("Vedantu classes fee INR 4500 TXNt2", "education"),
    ("PhysicsWallah batch INR 3600 TXNu3", "education"),
    ("Simplilearn certification INR 22000 TXNv4", "education"),
    ("Kuvera portfolio invest INR 12000 TXNw5", "investment"),
    ("Sukanya Samriddhi deposit INR 25000 TXNx6", "investment"),
    ("Bajaj Finserv EMI debit INR 3200 TXNy7", "emi"),
    ("HDFC auto loan installment INR 9800 TXNz8", "emi"),
]

texts = [t for t, _ in holdout]
truth = [c for _, c in holdout]
hpreds = model.predict(texts)
honest_acc = accuracy_score(truth, hpreds)

print(f"HONEST accuracy on unseen merchants: {honest_acc * 100:.1f}%")
print("  (26 brands that appear in neither training file - this is the")
print("   number that actually tells you if the model works)\n")

wrong = [(t, p, a) for t, p, a in zip(texts, hpreds, truth) if p != a]
if wrong:
    print("Got these wrong:")
    for t, p, a in wrong:
        print(f"  {t[:40]:<42} said {p:<14} should be {a}")
    print()


# ---- 6. Save ---------------------------------------------------------
joblib.dump(model, "model.pkl")
print("Saved to model.pkl")
