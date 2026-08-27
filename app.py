"""
app.py - the SpendSense backend.

Three jobs:
  1. Take transactions the user uploads
  2. Ask the trained model what category each one is
  3. Save them, and add up the totals

Run with:  uvicorn app:app --reload
Then open: http://localhost:8000
"""

import csv
import io
import sqlite3
from datetime import datetime

import joblib
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="SpendSense")

# Load the trained model once, when the server starts.
# Loading it on every request would be slow and pointless.
model = joblib.load("model.pkl")

DB = "spendsense.db"


# ---- Database --------------------------------------------------------
def setup_database():
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT    NOT NULL,
            amount      REAL    NOT NULL,
            category    TEXT    NOT NULL,
            confidence  REAL,
            added_at    TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


setup_database()


def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row   # lets us read columns by name
    return conn


# ---- Helper: ask the model -------------------------------------------
def categorise(description):
    """Returns (category, confidence). Confidence is how sure the model is."""
    category = model.predict([description])[0]
    probabilities = model.predict_proba([description])[0]
    confidence = float(max(probabilities))
    return category, confidence


def extract_amount(text):
    """Pull the rupee amount out of a description like 'Uber ride INR 480 TXN12'."""
    parts = text.replace(",", "").split()
    for i, word in enumerate(parts):
        if word.upper() in ("INR", "RS", "RS."):
            if i + 1 < len(parts):
                try:
                    return float(parts[i + 1])
                except ValueError:
                    pass
    # fall back: any number in the text
    for word in parts:
        try:
            return float(word)
        except ValueError:
            continue
    return 0.0


# ---- Endpoints -------------------------------------------------------
class SingleTransaction(BaseModel):
    description: str
    amount: float | None = None


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/api/predict")
def predict_one(txn: SingleTransaction):
    """Categorise one transaction without saving it. Used by the try-it box."""
    category, confidence = categorise(txn.description)
    return {
        "description": txn.description,
        "category": category,
        "confidence": round(confidence, 3),
    }


@app.post("/api/transactions")
def add_transaction(txn: SingleTransaction):
    """Categorise one transaction and save it."""
    category, confidence = categorise(txn.description)
    amount = txn.amount if txn.amount is not None else extract_amount(txn.description)

    conn = get_db()
    conn.execute(
        "INSERT INTO transactions (description, amount, category, confidence, added_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (txn.description, amount, category, confidence, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

    return {"description": txn.description, "amount": amount,
            "category": category, "confidence": round(confidence, 3)}


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV of transactions. It needs a column with the description in it.
    Every row gets categorised and saved.
    """
    raw = await file.read()
    text = raw.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))

    if not reader.fieldnames:
        return {"error": "That file has no header row."}

    # Find the description column, whatever it happens to be called.
    desc_col = None
    for name in reader.fieldnames:
        if any(word in name.lower() for word in
               ("description", "text", "narration", "particular", "detail")):
            desc_col = name
            break
    if desc_col is None:
        desc_col = reader.fieldnames[0]

    # Find an amount column if there is one.
    amt_col = None
    for name in reader.fieldnames:
        if any(word in name.lower() for word in ("amount", "amt", "debit", "value")):
            amt_col = name
            break

    conn = get_db()
    saved = 0
    for row in reader:
        description = (row.get(desc_col) or "").strip()
        if not description:
            continue

        category, confidence = categorise(description)

        amount = None
        if amt_col:
            try:
                amount = float(str(row[amt_col]).replace(",", "").strip())
            except (ValueError, TypeError, AttributeError):
                amount = None
        if amount is None:
            amount = extract_amount(description)

        conn.execute(
            "INSERT INTO transactions (description, amount, category, confidence, added_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (description, abs(amount), category, confidence, datetime.now().isoformat()),
        )
        saved += 1

    conn.commit()
    conn.close()
    return {"saved": saved, "used_column": desc_col}


@app.get("/api/transactions")
def list_transactions(limit: int = 100):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM transactions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return {"transactions": [dict(r) for r in rows]}


@app.get("/api/summary")
def summary():
    """Total spend per category, plus the overall total."""
    conn = get_db()
    rows = conn.execute("""
        SELECT category,
               COUNT(*)   AS count,
               SUM(amount) AS total
        FROM transactions
        GROUP BY category
        ORDER BY total DESC
    """).fetchall()
    total_row = conn.execute(
        "SELECT COUNT(*) AS n, SUM(amount) AS t FROM transactions"
    ).fetchone()
    conn.close()

    return {
        "by_category": [dict(r) for r in rows],
        "total_transactions": total_row["n"] or 0,
        "total_spend": total_row["t"] or 0,
    }


@app.get("/api/low-confidence")
def low_confidence(threshold: float = 0.5):
    """
    Transactions the model was unsure about.
    Worth showing the user so they can correct them.
    """
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM transactions WHERE confidence < ? ORDER BY confidence ASC LIMIT 50",
        (threshold,),
    ).fetchall()
    conn.close()
    return {"uncertain": [dict(r) for r in rows]}


@app.delete("/api/transactions")
def clear_all():
    conn = get_db()
    conn.execute("DELETE FROM transactions")
    conn.commit()
    conn.close()
    return {"cleared": True}
