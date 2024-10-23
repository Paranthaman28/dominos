Pizza Sales Forecasting and Ingredient Requirement Report
1. Introduction
The aim of this project is to forecast pizza sales and calculate the necessary ingredients required to fulfill the predicted demand. This is essential for inventory management and optimizing supply chain operations in a pizza business. We employ two forecasting methods: SARIMA and LSTM, both of which leverage historical sales data.

2. Data Description
Two datasets are utilized in this analysis:

Pizza Sales Dataset (Pizza_Sale - pizza_sales.csv):
Key Columns: order_date, order_time, quantity, total_price, pizza_name_id.
Pizza Ingredients Dataset (Pizza_ingredients - Pizza_ingredients.csv):
Key Columns: pizza_name_id, pizza_name, pizza_ingredients, Items_Qty_In_Grams.
3. Data Preprocessing
Converted the order_date to datetime format and filled any missing values.
Aggregated sales by date to create a daily sales time series.
Implemented log transformation on quantity to address positive skewness.
Identified outliers using the Interquartile Range (IQR) method.
Created additional flags for promotions and holidays to enrich the dataset.
4. Exploratory Data Analysis (EDA)
Visualizations were conducted to understand sales patterns, including:

Daily Sales Trend: A line plot showing total daily sales.
Sales by Month: Bar plots illustrating total sales quantity by month.
Sales by Pizza Size: Bar plots summarizing total sales by pizza size.
Sales by Pizza Category: Visual representation of total sales categorized by pizza types.
Sales by Day of the Week: Analysis of sales trends throughout the week.
5. Forecasting Models
Two forecasting models were implemented: SARIMA and LSTM.

5.1. SARIMA Model
The data was split into training and testing sets.
The SARIMA model was defined with parameters (p=1, d=1, q=1) and seasonal parameters (P=1, D=1, Q=1, S=7).
Forecasting was done for the next 7 days, yielding predicted values based on the test data.
5.2. LSTM Model
The quantity data was normalized using MinMaxScaler.
A sequence generator was created to prepare the input for the LSTM model.
The model architecture included two LSTM layers with Dropout for regularization.
Trained on 80% of the data, predictions were made on the remaining 20%.
6. Results
SARIMA Forecast: The predicted sales values for the next week were generated, allowing analysis of sales trends based on historical data.
LSTM Forecast: Similarly, the LSTM model provided forecasted values, which were inverse transformed to obtain actual sales figures.
7. Ingredient Requirement Calculation
Merged the predicted sales data with the ingredients dataset.
Calculated the total ingredient quantities required for each pizza type based on the predicted sales.
Compiled a purchase order detailing the total quantities required for each ingredient.
8. Conclusion
This project successfully forecasts pizza sales using SARIMA and LSTM models and provides a comprehensive ingredient purchase order based on predicted sales. Accurate forecasting aids in inventory management, minimizing waste, and ensuring that demand is met efficiently. Future work could focus on hyperparameter tuning for the models, as well as incorporating additional features such as marketing campaigns and competitor analysis.

9. References
Literature on time series forecasting using SARIMA and LSTM.
Documentation for the Python libraries used: Pandas, Keras, Statsmodels, Seaborn, Matplotlib.
Final Steps
Format the Report: Ensure it is well-structured with headings, subheadings, and visuals.
Visuals: Include graphs and plots generated in your script to support your findings.
Proofreading: Check for clarity and coherence, ensuring technical details are correct.
