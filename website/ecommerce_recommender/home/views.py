from django.shortcuts import render

import pandas as pd

def home(request):

    events = pd.read_csv("D:\GitHub\E-commerce-Product-Recommendation\data\processed\events_clean.csv")

    context = {
        "users": events["visitorid"].nunique(),
        "products": events["itemid"].nunique(),
        "interactions": len(events),
    }

    return render(request, "home/home.html", context)

def about(request):
    return render(request, "about.html")