from pathlib import Path

import pandas as pd
from django.core.paginator import Paginator
from django.shortcuts import render

# Adjust the parents[N] depth to match where this file actually sits
# relative to your project root (see earlier notes on this same issue
# in the recommendation view).
BASE_DIR = Path(__file__).resolve().parents[3]

# Loaded once when the server starts, not on every request.
_products_df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "products_labeled.csv"
)
_all_products = _products_df.to_dict("records")

PAGE_SIZE = 20


def products(request):

    paginator = Paginator(_all_products, PAGE_SIZE)
    page_number = request.GET.get("page", 1)
    page_obj = paginator.get_page(page_number)

    return render(
        request,
        "products/products.html",
        {"page_obj": page_obj},
    )