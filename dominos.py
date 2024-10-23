import pandas as pd
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.layers import LSTM, Dense, Dropout # type: ignore
from keras.models import Sequential # type: ignore

df_ing=pd.read_csv(r"C:\Users\Paranthaman\Downloads\Pizza_ingredients - Pizza_ingredients.csv")
df_ing

df_sales=pd.read_csv(r"C:\Users\Paranthaman\Downloads\Pizza_Sale - pizza_sales.csv")
df_sales

df_sales['order_date'] = pd.to_datetime(df_sales['order_date'], dayfirst=True, errors='coerce')
df_sales['order_time'] = pd.to_datetime(df_sales['order_time'], errors='coerce').dt.time
df_sales.fillna(method='ffill', inplace=True)
# Convert 'order_date' to datetime
df_sales['order_date'] = pd.to_datetime(df_sales['order_date'])

# Aggregate sales by date
daily_sales = df_sales.groupby('order_date')['total_price'].sum().reset_index()

def encoder(df_ing):
    le=LabelEncoder()
    for col in df_ing.columns:
        if df_ing[col].dtype == 'object':
            df_ing[col] = le.fit_transform(df_ing[col])
    return df_ing

def encoder(df_sales):
    le=LabelEncoder()
    for col in df_sales.columns:
        if df_sales[col].dtype == 'object':
            df_sales[col] = le.fit_transform(df_sales[col])
    return df_sales

# detecting the skewed columns using plot
def plot(df_sales,column):
  #distplot
  plt.figure(figsize=(15,4))
  plt.subplot(1,3,1)
  sns.distplot(df_sales[column])
  plt.title("distplot for"+" "+column)

  #histogram plot

  plt.subplot(1,3,2)
  sns.histplot(df_sales, x= column, kde= True, bins=30,color="salmon")
  plt.title("histogram plot for"+" "+column)

  #boxplot

  plt.subplot(1,3,3)
  sns.boxplot(df_sales, x=column)
  plt.title("Box plot for"+" "+column)


  #ther positive skewness so use log transformation

import numpy as np

# Apply log transformation (add a small constant to avoid log(0))
df_sales['log_quantity'] = np.log(df_sales['quantity']+1)  # Add 1 to avoid log(0)

def outlier_dec(df_sales):
    Q1_log = df_sales['log_quantity'].quantile(0.25)
    Q3_log = df_sales['log_quantity'].quantile(0.75)
    IQR_log = Q3_log - Q1_log

    # Calculate bounds
    lower_bound_log = Q1_log - 1.5 * IQR_log
    upper_bound_log = Q3_log + 1.5 * IQR_log

# Identify outliers
    outliers_log = df_sales[(df_sales['log_quantity'] < lower_bound_log) | (df_sales['log_quantity'] > upper_bound_log)]
    return outliers_log
#plot for sales date
def plots(df):
    plt.figure(figsize=(12, 6))
    plt.plot(daily_sales['order_date'], daily_sales['total_price'], marker='o', linestyle='-')
    plt.title('Daily Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Total Sales ($)')
    plt.grid()
    plt.show()
    return df
month_sales = df_sales.groupby('month')['quantity'].sum()
#plot sales with respect to month
def plots2(df_sales):
    plt.figure(figsize=(10,6))
    month_sales.plot(kind='bar', color='orange')
    plt.title('Total Sales by Month')
    plt.xlabel('Month')
    plt.ylabel('Total Quantity Sold')
    plt.xticks(ticks=range(1,13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.show()
    return df_sales
#plot of sale respect to size of pizza
size_sales = df_sales.groupby('pizza_size')['total_price'].sum().reset_index()
def plot3(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='pizza_size', y='total_price', data=size_sales, palette='Blues_d')
    plt.title('Total Sales by Pizza Size')
    plt.xlabel('Pizza Size')
    plt.ylabel('Total Sales ($)')
    plt.grid(axis='y')
    plt.show()
    return df
#plot with respect to pizza categories and price
category_sales = df_sales.groupby('pizza_category')['total_price'].sum().reset_index()
def plot4(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='pizza_category', y='total_price', data=category_sales, palette='Set2')
    plt.title('Total Sales by Pizza Category')
    plt.xlabel('Pizza Category')
    plt.ylabel('Total Sales ($)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()
    return df


#plot with respect to pizza week and quantity

day_of_week_sales = df_sales.groupby('day_of_week')['quantity'].sum()

def plot5(df):
    plt.figure(figsize=(10,6))
    day_of_week_sales.plot(kind='bar')
    plt.title('Total Sales by Day of the Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Total Quantity Sold')
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.show()
    return df


# Step 1: Define Promotional Periods for 2015
# Example promotional periods:
# - New Year Sale: January 1, 2015
# - Summer Promotion: July 1 - July 7, 2015
# - Black Friday: November 27, 2015
# - Christmas Promotion: December 24 - December 25, 2015
def promotion(df_sales):
    new_year_sale = pd.to_datetime(['2015-01-01'])
    summer_promotion = pd.date_range(start='2015-07-01', end='2015-07-07')
    black_friday = pd.to_datetime(['2015-11-27'])
    christmas_promotion = pd.date_range(start='2015-12-24', end='2015-12-25')

    # Combine all promotion periods into a single list of dates
    promotion_dates = new_year_sale.append(summer_promotion).append(black_friday).append(christmas_promotion)

    # Step 2: Define Holiday Periods for 2015
    holidays = pd.to_datetime([
        '2015-01-01',  # New Year's Day
        '2015-02-14',  # Valentine's Day
        '2015-04-05',  # Easter
        '2015-07-04',  # Independence Day
        '2015-11-26',  # Thanksgiving
        '2015-12-25'   # Christmas
    ])

    # Step 3: Add promotion and holiday flags to the dataset
    df_sales['is_promotion'] = df_sales['order_date'].isin(promotion_dates).astype(int)
    df_sales['is_holiday'] = df_sales['order_date'].isin(holidays).astype(int)
    return df_sales
def sarima_model():
# Split data into training and test sets (last 1 week for testing)
    train_data = day_of_week_sales[:7]
    test_data = day_of_week_sales[7:]

    # Define the SARIMA model (example parameters: p=1, d=1, q=1, seasonal: P=1, D=1, Q=1, S=7)
    sarima_model = SARIMAX(train_data, 
                        order=(1, 1, 1), 
                        seasonal_order=(1, 1, 1, 7), 
                        enforce_stationarity=False, 
                        enforce_invertibility=False)

    # Fit the model
    sarima_fit = sarima_model.fit(disp=False)

    # Forecast for the test period (7 days)
    forecast = sarima_fit.get_forecast(steps=7)
    forecast_values = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()

    # Predict sales for the next 7 days
    next_week_forecast = sarima_fit.get_forecast(steps=7)
    next_week_sales = next_week_forecast.predicted_mean
    return next_week_sales

    # Feature scaling for LSTM
def lstm(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_sales[['quantity']])

    # Define sequence generator for LSTM
    def create_sequences(data, seq_length):
        sequences = []
        labels = []
        for i in range(seq_length, len(data)):
            sequences.append(data[i-seq_length:i])
            labels.append(data[i])
        return np.array(sequences), np.array(labels)


    seq_length = 7  # Using past 7 days to predict the next day
    X, y = create_sequences(scaled_data, seq_length)

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


    # Build LSTM model
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

    # Make predictions
    predicted_sales = model.predict(X_test)

    # Inverse transform to get actual values
    predicted_sales = scaler.inverse_transform(predicted_sales)
    y_test_actual = scaler.inverse_transform(y_test)
    # Calculate MAPE

    mape =np.mean(np.abs((y_test_actual - predicted_sales) / y_test_actual)) * 100
    #for caste
    # Create a list to hold the forecasted values
    # Get the last 'time_step' days for prediction
    last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)

    # Create a list to hold the forecasted values
    forecasted_sales = []

    # Forecast for the next 7 days
    for _ in range(7):
        next_day_prediction = model.predict(last_sequence)
        forecasted_sales.append(next_day_prediction[0, 0])  # Get the scalar value
        # Update the last_sequence with the new prediction
        # This ensures that we are maintaining the right shape
        next_day_prediction = next_day_prediction.reshape(1, 1, 1)

        last_sequence = np.concatenate((last_sequence[:, 1:, :], next_day_prediction), axis=1)

    # Inverse transform the predictions to the original scale
    forecasted_sales = scaler.inverse_transform(np.array(forecasted_sales).reshape(-1, 1))

    # Display the forecasted sales
    last_date=df_sales.index[-1]
    if isinstance(last_date, pd.Timestamp):
        # Use last_date and add one day correctly
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)  # Start from the next day
    else:
        raise TypeError("The last_date is not a valid Timestamp.")
    return mape,forecast_dates

# Merge sales predictions with pizza ingredient data
def ing_pred():
    # Assuming we have a mapping between pizza_name_id and predicted_sales
    sales_data = df_sales[['pizza_name_id', 'quantity']]  # Pizza sales data from historical dataset

    # Group by pizza name and sum the predicted sales
    predicted_pizza_sales = sales_data.groupby('pizza_name_id').sum()

    # Merge with the ingredient data
    merged_data = pd.merge(predicted_pizza_sales, df_ing, on='pizza_name_id')

    # Calculate the required ingredients based on predicted sales
    merged_data['ingredient_qty_required'] = merged_data['quantity'] * merged_data['Items_Qty_In_Grams']
    merged_data# Final DataFrame with pizza and ingredients

    # Aggregate the total ingredients required across all pizzas
    purchase_order = merged_data.groupby('pizza_ingredients')['ingredient_qty_required'].sum().reset_index()

    # Rename columns for clarity
    purchase_order.columns = ['Ingredient', 'Total_Quantity_Required (Grams)']

    # Convert to kilograms if necessary (assuming 1000 grams = 1 kg)
    purchase_order['Total_Quantity_Required (Kilograms)'] = purchase_order['Total_Quantity_Required (Grams)'] / 1000

    # Display the purchase order
    return purchase_order,merged_data
