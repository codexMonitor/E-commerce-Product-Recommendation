from django.shortcuts import render

import pandas as pd

def products(request):

    df = pd.read_csv("D:\GitHub\E-commerce-Product-Recommendation\data\processed\products.csv")
    
    return render(request, "products/products.html", {"products": df.to_dict("records")})

    # return render(request, "products/products.html")
