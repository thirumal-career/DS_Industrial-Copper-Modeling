# Industrial Copper Modeling

**Introduction** : The objective of this project is to create two machine learning models specifically designed for the copper business. These models will be developed to tackle the difficulties associated with accurately anticipating selling prices and effectively classifying lead. The process of making forecasts manually may be a time-intensive task and may not provide appropriate price choices or effectively collect leads. The models will employ sophisticated methodologies, including data normalization, outlier detection and treatment, handling of improperly formatted data, identification of feature distributions, and utilization of tree-based models, particularly the decision tree algorithm, to accurately forecast the selling price and leads.

**Domain** : Manufacturing

## Prerequisites
1. **Python** -- Programming Language
2. **pandas** -- Python Library for Data Visualization
3. **numpy** --  Fundamental Python package for scientific computing in Python
4. **streamlit** -- Python framework to rapidly build and share beautiful machine learning and data science web apps
5. **scikit-learn** -- Machine Learning library for the Python programming language

<br/>

## Project Workflow
The following is a fundamental outline of the project:
  - This analysis aims to investigate the presence of skewness and outliers within the dataset.
  - The data will be converted into a format that is appropriate for analysis, and any required cleaning and pre-processing procedures will be carried out.
  - The objective of this study is to construct a machine learning regression model that utilizes the decision tree regressor to accurately forecast the continuous variable 'Selling_Price'.
  - The objective of this study is to construct a machine learning classification model using the decision tree classifier to accurately predict the outcome of a given task, namely whether it will result in a "WON" or "LOST" status.
  - The objective is to develop a Streamlit webpage that enables users to input values for each column and get the expected Selling_Price value or determine the Status (Won/Lost).

<br/>

## Using the App

### Selling Price Prediction
To predict the price of a copper transaction, follow these steps:
1. Select the **"Predict Selling Price"** tab.
2. Fill in the following required information:
   - Item Type
   - Status
   - Country
   - Application
   - Product Reference
   - Quantity in Tons
   - Thickness
   - Width
   - Customer ID
3. Click the **"Predict Selling Price"** button.
4. The app will display the predicted selling price based on the provided information.

### Status Prediction
To predict the status of a copper transaction, follow these steps:
1. Select the **"Predict Status"** tab.
2. Fill in the following required information:
   - Item Type
   - Country
   - Application
   - Product Reference
   - Quantity in Tons
   - Thickness
   - Width
   - Customer ID
   - Selling Price
4. Click the **"Predict Status"** button.
5. The app will display the predicted status for the copper transaction.
