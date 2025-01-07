## Repository Structure

├── README.md            
├── data/
│   └── products.csv     
├── src/
│   ├── analysis.py      
│   └── utils.py         
├── requirements.txt     
└── report.md            


## Install Dependencies:
The required Python packages are listed in requirements.txt. Run the following command to install:

```
pip install -r requirements.txt
```

### Configure Environment Variables:
The .env file should contain the API_KEY along with latitude and longitude of the location to fetch weather data:

```
LATITUDE=<latitude>
LONGITUDE=<longitude>
RAPIDAPI_KEY=<rapidapi_key>
```

## Running the Application
To run the application, execute the following command:

```
python src/analysis.py data/products.csv
```

This will:
1. Load the product data from the CSV file.
2. Train a text classification model to categorize the products.
3. Fetch the current weather data based on the latitude and longitude from the .env file.
4. Recommend products based on weather conditions (cold, hot, or snacks).
5. Suggest products that need to be restocked based on current weather conditions.

## Approach
1. Weather Information:
The weather data is fetched from the RapidAPI Weather API and classified into three categories based on weather conditions:
Cold: Products best suited for hot weather (e.g., cold beverages).
Hot: Products best suited for cold weather (e.g., hot drinks).
Snack: Neutral products that can be enjoyed in any weather.

2. Product Categorization Using Text Classification:
The system uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert product names into numerical vectors. This transformation helps the model assess the relevance of each product's name to different categories based on word frequency.
TF-IDF Vectorizer: Converts product names into vectors based on the frequency of terms in the product names, weighted by how rare or common those terms are across all product names.
Naive Bayes Classifier (MultinomialNB): The machine learning model that categorizes products into one of the following categories: hot, cold, or snack. The model is trained using product names and their corresponding categories.

3. Recommendations Based on Weather:
The system analyzes the current temperature and makes recommendations:
If the temperature is above 75°F: The recommendation will prioritize cold beverages and snacks.
If the temperature is below 50°F: The recommendation will focus on hot drinks and comforting snacks.
If the temperature is between 50°F and 75°F: All products are generally recommended, as the weather is considered moderate.

4. Restocking Based on Weather:
The system also checks the stock status of products and recommends which out-of-stock products should be restocked based on weather conditions.

5. Cold products should be restocked if the temperature is above 75°F.
Hot products should be restocked if the temperature is below 50°F.

6. The Weather API can be used by toggling the `USE_API_KEY` in the env file, set due to daily usage limits.
