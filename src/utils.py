import os

import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

load_dotenv()


def get_weather_data(lat, lon):
    """
    Use rapidAPI's "Weather API" to fetch local temperature based on coordinates
    """

    use_api_key = os.getenv("USE_API_KEY").strip().lower()
    if use_api_key == "true":
        url = "https://weatherbit-v1-mashape.p.rapidapi.com/current"

        querystring = {"lat": lat, "lon": lon, "units": "imperial", "lang": "en"}

        headers = {
            "x-rapidapi-key": os.getenv("RAPIDAPI_KEY"),
            "x-rapidapi-host": "weatherbit-v1-mashape.p.rapidapi.com",
        }

        response = requests.get(url, headers=headers, params=querystring)

        # check API response
        if response.status_code == 200:
            weather_data = response.json()
            temperature = weather_data["data"][0]["temp"]
            return temperature
        else:
            print(f"Error fetching weather data: {response.status_code}")
            return None
    # For local testing or if API limit reached
    else:
        temperature = 75
        return temperature


def train_product_classifier():
    """
    Trains a simple text classification model to categorize products as hot, cold, or snack.
    """
    product_names = [
        "Coffee",
        "Chai Latte",
        "Hot Chocolate",
        "Iced Coffee",
        "Green Tea",
        "Scones",
        "Cinnamon Rolls",
        "Croissants",
        "Bagels",
        "Muffins",
        "Chocolate Chip Cookies",
        "Brownies",
        "Lemonade",
        "Iced Tea",
        "Frappuccino",
        "Hot Apple Cider",
        "Mocha",
        "Caramel Macchiato",
        "Latte",
        "Espresso",
        "Iced Latte",
        "Pumpkin Spice Latte",
        "Matcha Latte",
        "Peppermint Mocha",
        "Apple Cider",
        "Lemon Iced Tea",
        "Cold Brew Coffee",
        "Tea with Honey",
        "Vegan Muffin",
        "Blueberry Muffins",
        "Cheese Croissant",
        "Ice Cream",
        "Almond Biscotti",
        "Fruit Salad",
    ]

    categories = [
        "hot",
        "hot",
        "hot",
        "cold",
        "cold",
        "snack",
        "hot",
        "snack",
        "snack",
        "snack",
        "snack",
        "snack",
        "cold",
        "cold",
        "cold",
        "hot",
        "hot",
        "hot",
        "hot",
        "hot",
        "cold",
        "cold",
        "cold",
        "snack",
        "snack",
        "snack",
        "snack",
        "snack",
        "snack",
        "snack",
        "snack",
        "cold",
        "snack",
        "hot",
    ]

    assert len(product_names) == len(
        categories
    ), "The number of products must match the number of categories"

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(product_names)
    model = MultinomialNB()
    model.fit(X, categories)

    return model, vectorizer


def categorize_product(model, vectorizer, product_name):
    """
    Categorizes a product into hot, cold, or snack using the trained model.
    """
    X_new = vectorizer.transform([product_name])
    return model.predict(X_new)[0]


def get_weather_recommendations(weather_data, product_data, model, vectorizer):
    """
    Makes recommendations based on weather and product category.
    Only high demand products are included.
    """
    if not weather_data:
        print("No weather data available for recommendations.")
        return []

    temperature = weather_data
    high_demand_items = []

    for index, row in product_data.iterrows():
        product_name = row["product_name"]
        category = categorize_product(model, vectorizer, product_name)

        if temperature > 75:
            if category == "cold":
                high_demand_items.append(product_name)

        elif temperature < 50:
            if category == "hot":
                high_demand_items.append(product_name)

        # don't fluctuate prices in moderate weather
        elif 50 <= temperature <= 75:
            high_demand_items.append(product_name)

    return high_demand_items


def check_and_restock(weather_data, product_data, high_demand_items):
    """
    Based on the weather data, suggest which out-of-stock products should be restocked.
    """
    if not weather_data:
        print("No weather data available for restocking.")
        return []

    restock_items = []

    for index, row in product_data.iterrows():
        product_name = row["product_name"]
        stock_status = row["stock_status"].strip().lower()

        if product_name not in high_demand_items:
            continue
        if stock_status == "out-of-stock":
            restock_items.append(product_name)
    return restock_items
