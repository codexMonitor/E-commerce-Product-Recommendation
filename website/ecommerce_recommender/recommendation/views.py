from pathlib import Path
import sys
import pickle
import pandas as pd

from django.conf import settings
from django.shortcuts import render

BASE_DIR = Path(__file__).resolve().parents[3]

sys.path.append(str(BASE_DIR))

sys.path.append(str(BASE_DIR / "notebooks"))

from recommender_wrapper import Recommender

# ----------------------------
# Load packaged recommender + product catalogue
# ----------------------------

try:
    with open(BASE_DIR / "models" / "recommender.pkl", "rb") as f:
        recommender = pickle.load(f)

    products = pd.read_csv(
        BASE_DIR / "data" / "processed" / "products_labeled.csv"
    )

    products = products.drop_duplicates(subset="itemid", keep="last")

    STARTUP_ERROR = None

except Exception as e:
    recommender = None
    products = None
    STARTUP_ERROR = str(e)


def recommendation(request):

    recommendations = []
    error = STARTUP_ERROR

    if request.method == "POST" and error is None:

        try:

            visitor_id = int(request.POST["user"])
            top_k = int(request.POST["k"])

            # Wrapper handles everything
            results = recommender.recommend(
                visitor_id=visitor_id,
                top_k=top_k
            )

            if len(results) == 0:
                error = "No recommendations found."

            else:

                score_map = {
                    r["item_id"]: r["score"]
                    for r in results
                }

                item_ids = list(score_map.keys())

                recommendations = (
                    products[
                        products["itemid"].isin(item_ids)
                    ]
                    .copy()
                )

                recommendations["score"] = (
                    recommendations["itemid"].map(score_map)
                )

                recommendations = recommendations.dropna(subset=["score"])

                recommendations = recommendations.sort_values(
                    "score",
                    ascending=False
                )

                # Respect top_k even if isin() matched extra rows
                recommendations = recommendations.head(top_k)

                recommendations = recommendations.to_dict("records")

        except Exception as e:
            error = str(e)

    return render(
        request,
        "recommendation/recommendation.html",
        {
            "recommendations": recommendations,
            "error": error,
        },
    )