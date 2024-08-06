# Maize-Prices-Prediction
# README
Youtube Link Demo : https://youtu.be/YJeiL7QeGpI
## Maize Price Prediction Application

This README file provides a comprehensive guide to hosting a Maize Price Prediction application on a local server using Streamlit. The application leverages machine learning models to predict maize prices based on features such as year, month, county, and region.

### Table of Contents

1. [Overview](#overview)
2. [Packages and Libraries](#packages-and-libraries)
3. [Setup and Installation](#setup-and-installation)
4. [Data Preparation](#data-preparation)
5. [Model Training](#model-training)
6. [Application Deployment](#application-deployment)
7. [Running the Application](#running-the-application)

### Overview

The Maize Price Prediction application is built using Python and Streamlit. The application allows users to input specific features (year, month, county, and region) and predicts the maize prices based on a pre-trained RandomForestRegressor model. This application was developed with the following steps:

1. **Data Preparation**: Clean and preprocess the data.
2. **Model Training**: Train multiple regression models and select the best-performing one.
3. **Application Development**: Build a Streamlit application for user interaction.
4. **Deployment**: Host the application on a local server.

### Packages and Libraries

The following packages and libraries were used in the project:

- pandas
- numpy
- scikit-learn
- joblib
- matplotlib
- streamlit

### Setup and Installation

To set up the environment and install the necessary packages, follow these steps:

1. **Install Python**: Ensure you have Python installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).

2. **Create a Virtual Environment**:
   ```bash
   python -m venv maize_price_env
   ```

3. **Activate the Virtual Environment**:
   - On Windows:
     ```bash
     .\maize_price_env\Scripts\activate
     ```

4. **Install Required Packages**:
   ```bash
   pip install pandas numpy scikit-learn joblib matplotlib streamlit
   ```

### Data Preparation

Ensure your dataset is properly formatted and cleaned. For this example, let's assume the dataset is in a DataFrame called `train_df` with a target column `Price`.

### Model Training

The following script trains multiple regression models, evaluates their performance, and saves the best model:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt

# Assuming train_df is already created

# Define your features and target
features = train_df.drop(columns=['Price'])  # replace 'Price' with your actual target column name if different
target = train_df['Price']

# Identify categorical columns
categorical_cols = features.select_dtypes(include=['object', 'category']).columns.tolist()

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode each categorical column
for column in categorical_cols:
    features[column] = label_encoder.fit_transform(features[column])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize models
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'LinearRegression': LinearRegression(),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate each model
best_model_name = None
best_model = None
best_score = float('-inf')

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'{model_name} - Mean Squared Error: {mse}, R2 Score: {r2}')
    
    # Save the best model based on R2 score
    if r2 > best_score:
        best_score = r2
        best_model_name = model_name
        best_model = model

    # Create a DataFrame to display the first 10 actual vs predicted values
    comparison_df = pd.DataFrame({
        'Actual Prices': y_test[:10].values,
        'Predicted Prices': y_pred[:10].flatten()
    })

    print("First 10 actual vs predicted values:")
    print(comparison_df)

# Save the best model
if best_model:
    joblib.dump(best_model, 'best_model.pkl')
    print(f'The best model ({best_model_name}) has been saved.')

# Visualize the actual vs predicted prices using the best model
y_pred = best_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Prediction')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('RandomForest - Actual vs Predicted Prices')
plt.legend()
plt.show()
```

### Application Deployment

The following Streamlit application uses the trained model to predict maize prices based on user inputs:

```python
import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("best_model.pkl")

# Define regions and counties
regions = ['Coast', 'Eastern', 'Nairobi', 'North Eastern', 'Nyanza', 'Rift Valley', 'Western']
county_region_map = {
    'Coast': ['Mombasa', 'Kwale', 'Kilifi', 'Tana River', 'Lamu', 'Taita-Taveta'],
    'Eastern': ['Machakos', 'Makueni', 'Kitui', 'Embu', 'Meru', 'Isiolo', 'Tharaka-Nithi'],
    'Nairobi': ['Nairobi'],
    'North Eastern': ['Garissa', 'Wajir', 'Mandera'],
    'Nyanza': ['Kisumu', 'Siaya', 'Homa Bay', 'Migori', 'Kisii', 'Nyamira'],
    'Rift Valley': ['Uasin Gishu', 'Elgeyo-Marakwet', 'Nandi', 'Baringo', 'Laikipia', 'Nakuru', 'Narok', 'Kajiado', 'Kericho', 'Bomet', 'Samburu', 'Turkana', 'West Pokot'],
    'Western': ['Kakamega', 'Vihiga', 'Bungoma', 'Busia'],
}

# Streamlit app title
st.title('Maize Price Prediction')

# User input widgets
region = st.selectbox('Select Region', regions)
county = st.selectbox('Select County', county_region_map[region])
year = st.slider('Select Year', 2024, 2030, 2025)
month = st.selectbox('Select Month', list(range(1, 13)))

# Function to prepare input data for prediction
def prepare_input_data(region, county, year, month):
    input_data = {
        'Year': year,
        'Month': month,
        'County': county,
        'Region': region
    }

    # Encode categorical columns
    categorical_cols = ['Region', 'County']
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit([region, county])
        input_data[col] = le.transform([input_data[col]])[0]

    return pd.DataFrame([input_data])

# Predict maize prices when the button is clicked
if st.button('Predict Price'):
    input_df = prepare_input_data(region, county, year, month)
    st.write("Input data for prediction:")
    st.write(input_df)  # Debugging: Display input data
    
    try:
        predicted_price = model.predict(input_df)[0]
        st.write(f'Predicted Price for {county}, {region} in {year}-{month}: {predicted_price:.2f} KES')
    except Exception as e:
        st.error(f"Error during prediction: {e}")
```

### Running the Application

To run the Streamlit application, use the following command in your terminal:

```bash
streamlit run app.py
```

This command will start the Streamlit server and open a new tab in your default web browser, where you can interact with the Maize Price Prediction application.

