# E-commerce Product Recommendation

**DS 423 — Machine Learning with Large Datasets**
Duy Tan University, Da Nang — Group 7

A full-stack implicit-feedback recommender system built on the [Retailrocket
e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset),
trained with Alternating Least Squares (ALS) matrix factorization and served
through a Django web application.

## Team

| Member | Student ID | Role |
|---|---|---|
| Trần Bảo Duy | 28211354517 | Data & Analysis Lead — data pipeline, EDA, Django web application |
| Trương Đức Quốc Huy | 29211154415 | Modeling & Evaluation Lead — ALS training, hyperparameter tuning, evaluation |

## Overview

Retail websites accumulate large volumes of implicit user behavior — views,
add-to-cart actions, and purchases — that can be used to recommend products
without ever asking a customer to rate anything directly. This project turns
that raw event log into a working recommendation pipeline:

1. **Preprocess** raw Retailrocket events into a weighted user–item
   interaction matrix.
2. **Train** an ALS collaborative filtering model (`implicit` library) on
   that matrix.
3. **Evaluate** the model against a popularity baseline using Precision@K,
   Recall@K, and NDCG.
4. **Serve** the trained model through a Django app, where a visitor ID can
   be submitted to get ranked product recommendations, and the full product
   catalog can be browsed with pagination.

## Dataset

The [Retailrocket dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
contains anonymized events from a real online store:

- `events.csv` — user interaction events (`view`, `addtocart`, `transaction`)
- `item_properties.csv` (parts 1 & 2) — time-varying item metadata, including category
- `category_tree.csv` — hierarchical category relationships

After cleaning (deduplication, re-indexing IDs):

| Statistic | Value |
|---|---|
| Total raw events | 2,756,101 |
| Unique visitors (in interaction matrix) | 1,407,579 |
| Unique items with category metadata | 417,053 |
| Unique items with at least one event | 235,061 |
| Views / Add-to-cart / Transactions | 2,664,312 / 69,332 / 22,457 |

## Model

- **Algorithm:** Alternating Least Squares (ALS) for implicit feedback, via the [`implicit`](https://github.com/benfred/implicit) library
- **Hyperparameter search:** grid search over factors (20/50/100), regularization (0.01/0.05/0.1), and iterations (10/20) — 18 configurations, selected by NDCG@10
- **Best configuration:** `factors=100`, `regularization=0.05`, `iterations=20`
- **Train/test split:** 80/20 (`random_state=42`) — 1,716,140 train / 429,036 test interactions

## Results

Evaluated on a random sample of 5,000 test users, compared against a
popularity baseline (most popular not-yet-interacted items):

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Popularity Baseline | 0.00092 | 0.00860 | 0.00439 |
| **ALS (tuned)** | **0.00318** | **0.02412** | **0.01576** |

The tuned ALS model outperforms the popularity baseline by roughly **3.5x**
on precision, **2.8x** on recall, and **3.6x** on NDCG — confirming it
captures genuine personalized signal rather than just surfacing best-sellers.

## Project Structure

```
├── notebooks/              # Data prep, EDA, model training & evaluation notebooks
├── models/                 # recommender_wrapper.py, packaged recommender.pkl
├── website/                # Django project (recommendation form, product catalog)
├── data/
│   ├── raw/                # Original Retailrocket CSVs
│   └── processed/          # Cleaned interaction matrix, encoders, products.csv
├── results/                 # Evaluation metrics and charts
├── requirements.txt
└── README.md
```

*The tree does not look the same as above because of some technical problems*

## Setup & Usage

```bash
# clone the repo
git clone https://github.com/codexMonitor/E-commerce-Product-Recommendation.git
cd E-commerce-Product-Recommendation

# install dependencies
pip install -r requirements.txt

# move to the folder named website before running the project
cd website/ecommerce_recommender

# run the Django web app
python manage.py runserver
```

Then open `http://127.0.0.1:8000/recommendation/` to get recommendations for
a given visitor ID, or `http://127.0.0.1:8000/products/` to browse the
paginated product catalog.

## Tech Stack

- **Modeling:** Python, pandas, NumPy, SciPy (sparse matrices), `implicit` (ALS)
- **Web application:** Django
- **Report:** LaTeX (XeLaTeX)

## References

- Zykov, R. (2016). *Retailrocket recommender system dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
- Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. *2008 Eighth IEEE International Conference on Data Mining*, 263–272. https://doi.org/10.1109/ICDM.2008.22
