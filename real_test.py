"""
real_test.py - the honest test.

The Kaggle test set reuses the same 45 phrases as the training set, so the
model scores 100% just by memorising. That number is meaningless.

The real question is: does it work on merchants it has NEVER seen?
This file checks exactly that.
"""

import joblib

model = joblib.load("model.pkl")

# Merchants that appear NOWHERE in the training data.
unseen = [
    # (transaction text, the correct answer)
    ("Dominos pizza order INR 890 TXN44a1", "food"),
    ("BigBasket grocery INR 2340 TXN88c2", "food"),
    ("Blinkit delivery INR 450 TXN1f39", "food"),
    ("Starbucks coffee INR 520 TXN7b21", "food"),

    ("Rapido bike ride INR 90 TXN22de", "travel"),
    ("IndiGo flight INR 6700 TXN9a04", "travel"),
    ("Petrol pump HP INR 2000 TXN3c71", "travel"),
    ("MakeMyTrip booking INR 12400 TXN5e88", "travel"),

    ("Myntra fashion order INR 1899 TXN6d10", "shopping"),
    ("Ajio clothing INR 2299 TXN0b45", "shopping"),
    ("Croma electronics INR 34999 TXNa917", "shopping"),
    ("Nykaa beauty INR 1150 TXN4f62", "shopping"),

    ("Airtel postpaid payment INR 799 TXN8e13", "utilities"),
    ("Jio fiber broadband INR 999 TXNc204", "utilities"),
    ("Tata Power bill INR 3400 TXN71ab", "utilities"),

    ("Spotify premium INR 199 TXN9d55", "entertainment"),
    ("BookMyShow tickets INR 700 TXN2a83", "entertainment"),
    ("PVR cinemas INR 850 TXNe419", "entertainment"),

    ("Apollo pharmacy INR 640 TXN5b27", "healthcare"),
    ("PharmEasy medicines INR 1230 TXN8f01", "healthcare"),
    ("Dental clinic charges INR 3500 TXN6c94", "healthcare"),

    ("Udemy course purchase INR 499 TXN1d76", "education"),
    ("Coursera subscription INR 3999 TXNb832", "education"),
    ("Tuition fees payment INR 15000 TXN4e20", "education"),

    ("Zerodha stock buy INR 25000 TXN7a58", "investment"),
    ("Groww mutual fund INR 5000 TXN3b19", "investment"),
    ("NPS contribution INR 50000 TXNd671", "investment"),

    ("Two wheeler loan payment INR 4500 TXN9c33", "emi"),
    ("Education loan installment INR 12000 TXN0a87", "emi"),
]

texts = [t for t, _ in unseen]
truth = [c for _, c in unseen]
predictions = model.predict(texts)

correct = 0
print(f"{'TRANSACTION':<42} {'PREDICTED':<15} {'ACTUAL':<15}")
print("-" * 78)

for text, pred, actual in zip(texts, predictions, truth):
    mark = "OK " if pred == actual else "X  "
    if pred == actual:
        correct += 1
    short = text[:40]
    print(f"{mark}{short:<39} {pred:<15} {actual:<15}")

print("-" * 78)
print(f"\nAccuracy on UNSEEN merchants: {correct}/{len(unseen)} = {correct/len(unseen)*100:.1f}%")
