# 🏍️ Used Bike Price Prediction

A Flask web app that estimates the resale price (₹) of a used bike from
its brand, number of previous owners, age, engine power (cc) and
kilometres driven, using a Linear Regression model trained on 7,000+
used-bike listings.

## What it does

Pick the brand (23 brands: Royal Enfield, KTM, Bajaj, Honda, Hero, TVS,
Yamaha, Kawasaki, Ducati, BMW, Harley-Davidson, and more), owner number,
age, power and kms driven — the app estimates the price, echoes the bike's
details, and keeps a session history of your predictions (History page,
linked from the home page).

## ⚠️ Important fixes in this version

1. **Feature order corrected.** The model was trained on
   `[owner, brand, kilometer, age, power]`, but the old app sent
   `[brand, owner, age, power, kms]` — every prediction used scrambled
   inputs.
2. **Brand codes corrected.** The old app's hand-written brand→code
   dictionary was shuffled (e.g. it called code 17 "Indian"). The codes
   are the LabelEncoder (alphabetical) codes from data cleaning —
   verified against the dataset itself (code 17 = 350cc, ~₹1.1 lakh,
   most common premium brand = Royal Enfield; code 1 = the most common
   150cc commuter brand = Bajaj, etc.).
3. **The working model is now used.** The app previously loaded
   `model.lb` (a Random Forest from a different dataset version — it
   scores **R² < 0** on the bundled data). It now loads
   `ml_used_bikes.lb`, the model trained in the bundled notebook on the
   bundled dataset.
4. **Prediction crash fixed** (2-D prediction output handled), the
   Harley-Davidson form option (whose value never matched) fixed, the
   missing Hero and Jawa options added, and the `/history` page — linked
   from the home page but previously returning 404 — now works.

## Verified model performance

- **Test R² = 0.6954** (Linear Regression, 80/20 train–test split,
  `random_state=42`, reproduced from `ml_used_bikes.ipynb`; the saved
  model scores identically)

## Tech stack

- Python, Flask
- scikit-learn (LinearRegression), pandas, NumPy, joblib

## How to run

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 and go to the prediction page.

## Files

- `app.py` — Flask routes (`/`, `/project`, `/about`, `/contact`,
  `/history`, `/health`)
- `ml_used_bikes.lb` — trained Linear Regression model (used by the app)
- `model.lb` — older Random Forest model file (kept for reference; not
  used — see fix #3 above)
- `ml_used_bikes.ipynb` — training notebook
- `new_Cleaned_Used_Bikes.csv` — cleaned dataset (7,324 listings)

## Dataset note

Used-bike listings (owner, brand, price, kilometres driven, age, engine
power) — the popular Kaggle used-bikes dataset.
