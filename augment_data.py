"""
augment_data.py - fixes the weakness we found.

The Kaggle set has only 45 phrases, so the model memorises instead of
learning. Here we add many more real Indian merchants per category so the
model sees enough variety to generalise to brands it hasn't met.

This is called data augmentation. It is the standard fix when a model
does well on its test set but fails on new inputs.
"""

import csv
import random

random.seed(42)

MERCHANTS = {
    "food": [
        "Swiggy order", "Zomato dinner", "Pizza delivery", "Restaurant bill",
        "Lunch at hotel", "Cafe coffee", "Food court payment", "Dominos pizza",
        "BigBasket grocery", "Blinkit delivery", "Zepto groceries", "Starbucks coffee",
        "McDonalds meal", "KFC order", "Burger King", "Subway sandwich",
        "Haldiram sweets", "Barbeque Nation", "Cafe Coffee Day", "Dunzo food",
        "Licious meat order", "Grofers online", "Bakery purchase", "Milk dairy payment",
        "Canteen payment", "Tea stall", "Biryani order", "Chinese takeaway",
        "Grocery store bill", "Supermarket purchase", "Fruit vendor", "Sweet shop",
    ],
    "travel": [
        "Uber ride", "Ola cab", "Bus ticket", "Train ticket IRCTC",
        "Flight ticket booking", "Hotel booking", "Rapido bike ride", "IndiGo flight",
        "SpiceJet airlines", "Air India ticket", "MakeMyTrip booking", "Goibibo hotel",
        "RedBus ticket", "Metro card recharge", "Petrol pump HP", "Indian Oil fuel",
        "Bharat Petroleum fuel", "Toll plaza payment", "FASTag recharge", "Parking fee",
        "Cab fare payment", "Auto rickshaw", "Yatra travel booking", "Railway reservation",
        "Airport taxi", "Car rental", "Diesel purchase",
    ],
    "shopping": [
        "Amazon purchase", "Flipkart order", "Online shopping payment", "Clothing store bill",
        "Electronics shopping", "Myntra fashion order", "Ajio clothing", "Nykaa beauty",
        "Croma electronics", "Reliance Digital", "Meesho order", "Tata Cliq purchase",
        "Snapdeal buy", "Shoppers Stop", "Lifestyle store", "Pantaloons purchase",
        "Decathlon sports", "IKEA furniture", "Mall purchase", "Footwear shop",
        "Jewellery purchase", "Furniture order", "Cosmetics purchase", "Bookstore payment",
        "Stationery purchase", "Gift shop",
    ],
    "utilities": [
        "Electricity bill payment", "Water bill", "Gas bill payment", "Mobile recharge",
        "Internet recharge", "Airtel postpaid payment", "Jio fiber broadband", "Vi prepaid recharge",
        "ACT Fibernet bill", "Tata Power bill", "BSNL landline", "DTH recharge",
        "LPG cylinder booking", "Municipal tax payment", "Broadband payment", "Maintenance charges",
        "Society maintenance", "Sewage charges", "Postpaid bill payment",
    ],
    "entertainment": [
        "Netflix subscription", "Movie ticket booking", "Concert ticket", "Game purchase",
        "Spotify premium", "BookMyShow tickets", "PVR cinemas", "INOX movies",
        "Amazon Prime subscription", "Hotstar subscription", "JioCinema plan", "YouTube Premium",
        "Steam games purchase", "PlayStation store", "Theme park ticket", "Bowling alley",
        "Music streaming", "OTT subscription", "Gaming top up", "Event ticket",
    ],
    "healthcare": [
        "Hospital bill", "Doctor consultation", "Pharmacy medicine purchase", "Medical test charges",
        "Apollo pharmacy", "PharmEasy medicines", "1mg medicine order", "MedPlus store",
        "Dental clinic charges", "Eye checkup", "Lab test diagnostic", "Practo consultation",
        "Physiotherapy session", "Health insurance premium", "Ambulance charges", "Vaccination fee",
        "Gym membership", "Cult Fit subscription", "Surgery payment", "Clinic visit",
    ],
    "education": [
        "College fees payment", "Exam fee", "Library fee", "Online course subscription",
        "Udemy course purchase", "Coursera subscription", "Tuition fees payment", "School fees",
        "Byjus subscription", "Unacademy plan", "Textbook purchase", "Hostel fees",
        "Coaching class fee", "Certification exam fee", "University fee", "Semester fee",
        "Workshop registration", "Training program fee", "Study material purchase",
    ],
    "investment": [
        "Mutual fund SIP", "Stock purchase", "Crypto investment", "Fixed deposit investment",
        "PPF contribution", "Zerodha stock buy", "Groww mutual fund", "NPS contribution",
        "Upstox trading", "Recurring deposit", "Gold bond purchase", "ELSS investment",
        "Bond purchase", "SIP installment", "Demat account funding", "Sovereign gold bond",
        "Index fund purchase", "Equity investment",
    ],
    "emi": [
        "Home loan EMI", "Car loan EMI", "Bike loan EMI", "Credit card EMI",
        "Personal loan installment", "Two wheeler loan payment", "Education loan installment",
        "Consumer durable EMI", "Mobile phone EMI", "Laptop EMI payment", "Gold loan EMI",
        "Business loan installment", "Loan repayment", "Monthly EMI debit",
        "Housing loan payment", "Vehicle loan EMI",
    ],
}


def build_rows(per_merchant=6):
    """Make several variations of each merchant, with random amounts and IDs."""
    rows = []
    for category, merchants in MERCHANTS.items():
        for merchant in merchants:
            for _ in range(per_merchant):
                amount = random.randint(50, 50000)
                txn = "TXN" + "".join(random.choices("0123456789abcdef", k=8))
                text = f"{merchant} INR {amount} {txn}"
                rows.append((text, category))
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = build_rows()

    with open("augmented_train.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["transaction_text", "category"])
        writer.writerows(rows)

    total_merchants = sum(len(m) for m in MERCHANTS.values())
    print(f"Merchants covered: {total_merchants}  (Kaggle set had 45)")
    print(f"Rows written: {len(rows)}")
    print("Saved to augmented_train.csv")
