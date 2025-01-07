import os
import sys

import pandas as pd
from dotenv import load_dotenv

from utils import (
    check_and_restock,
    get_weather_data,
    get_weather_recommendations,
    train_product_classifier,
)

load_dotenv()


def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        sys.exit(f"Error reading the CSV file: {e}")


def write_to_report(report_file, text):
    """Helper function to write text to the report file"""
    with open(report_file, "a") as file:
        file.write(text + "\n")


def main(file_path, lat, lon, report_file):
    # Initialize report.md
    write_to_report(
        report_file, "# Product Categorization and Weather Analysis Report\n"
    )

    # Get csv data
    data = load_data(file_path)

    write_to_report(report_file, "## Key Insights Discovered\n")

    write_to_report(report_file, "Training product categorization model...")

    model, vectorizer = train_product_classifier()

    # weather information
    write_to_report(report_file, "Fetching weather data...")

    temperature = get_weather_data(lat, lon)
    if temperature is not None:
        write_to_report(report_file, f"Current temperature: {temperature}°F\n")

        # Weather-based recommendation system
        high_demand_items = get_weather_recommendations(
            temperature, data, model, vectorizer
        )

        if temperature > 75:
            write_to_report(
                report_file,
                "\nBeat the heat with our refreshing cold drinks and tasty snacks!",
            )
        elif temperature < 50:
            write_to_report(
                report_file,
                "\nBeat the chill with something warm from our hot drinks and comforting snacks!",
            )
        else:
            write_to_report(
                report_file,
                "\nEnjoy a wide range of drinks and snacks for this perfect weather!",
            )

        write_to_report(report_file, "\nHigh Demand Products Based on Weather:")
        for item in high_demand_items:
            write_to_report(report_file, f"- {item}")

        # restock out-of-stock products based on weather
        restock_items = check_and_restock(temperature, data, high_demand_items)
        if restock_items:
            write_to_report(
                report_file,
                "\nThese products should be restocked based on the weather conditions:",
            )
            for item in restock_items:
                write_to_report(report_file, f"- {item}")
        else:
            write_to_report(
                report_file,
                "\nNo products need to be restocked for the current weather conditions.",
            )
    else:
        write_to_report(
            report_file, "Could not retrieve weather data for recommendations."
        )

    # Final Insights
    write_to_report(report_file, "## Recommendations Based on Findings\n")
    write_to_report(
        report_file,
        "- Use the 5-day forecast API to define promotion strategies for customers.\n",
    )
    write_to_report(report_file, "- Set up promotional offers for regular customers.\n")
    write_to_report(
        report_file, "- Monitor stock levels regularly and restock items accordingly.\n"
    )

    write_to_report(report_file, "## External Data Source\n")
    write_to_report(
        report_file, "- Weather data sourced from Weatherbit API (RapidAPI).\n"
    )


if __name__ == "__main__":
    file_path = sys.argv[1]  # CSV file path
    # Load coordinates for weather, from the .env file
    lat = float(os.getenv("LATITUDE"))
    lon = float(os.getenv("LONGITUDE"))

    # Specify the report file path
    report_file = "report.md"

    # Clear the file before writing (optional)
    open(report_file, "w").close()

    # Main function to train model, get weather data, display favorable drinks/snacks, and check for restocking
    main(file_path, lat, lon, report_file)
