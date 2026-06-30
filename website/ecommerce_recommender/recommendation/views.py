from django.shortcuts import render

def recommendation(request):

    recommendations = None

    if request.method == "POST":

        recommendations = [

            "Product 101",

            "Product 205",

            "Product 399",

        ]

    return render(

        request,

        "recommendation/recommendation.html",

        {"recommendations": recommendations}

    )
