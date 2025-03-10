# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime as dt
import matplotlib.pyplot as plt
import time  # For timer

# Set plot style
plt.style.use('fivethirtyeight')

# Streamlit app title
st.title("Stock Price Prediction with LSTM")

# User inputs for stock ticker and future date
stock = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, POWERGRID.NS)", "AAPL")
future_date = st.date_input("Enter a Future Date for Prediction", dt.date.today() + dt.timedelta(days=30))

# Button to trigger prediction
if st.button("Predict Future Price"):
    # Fetch historical stock data from Yahoo Finance
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=365 * 5)  # 5 years of historical data
    df = yf.download(stock, start=start_date, end=end_date)

    if df.empty:
        st.error("No data found for the given stock ticker. Please check the ticker symbol.")
    else:
        # Display fetched data preview
        st.write(f"Historical Data for {stock}:", df.tail())

        # Extract only the 'Close' price column for prediction
        data = df[['Close']].values

        # Normalize the data using MinMaxScaler (0-1 range)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Prepare training data (use 80% of the data for training)
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]

        # Function to create sequences of 60 days to predict the next day's price
        def create_sequences(dataset, time_step=60):
            x, y = [], []
            for i in range(time_step, len(dataset)):
                x.append(dataset[i-time_step:i, 0])
                y.append(dataset[i, 0])
            return np.array(x), np.array(y)

        # Create training sequences
        time_step = 60
        x_train, y_train = create_sequences(train_data, time_step)

        # Reshape input to be [samples, time steps, features] for LSTM
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
            Dropout(0.2),
            LSTM(60, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model with a progress bar and timer
        epochs = 20
        progress_bar = st.progress(0)
        status_text = st.empty()  # Placeholder for status updates

        start_time = time.time()  # Start timer

        for epoch in range(epochs):
            model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)  # Train for 1 epoch
            progress_bar.progress((epoch + 1) / epochs)  # Update progress bar
            status_text.text(f"Epoch {epoch + 1}/{epochs} completed | Time elapsed: {int(time.time() - start_time)} seconds")

        # Prepare test data (last 60 days of the training data)
        test_data = scaled_data[train_size - time_step:]
        x_test, y_test = create_sequences(test_data, time_step)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        # Predict on test data
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Predict future price
        last_60_days = scaled_data[-time_step:]  # Use the last 60 days of data
        last_60_days = last_60_days.reshape(1, time_step, 1)
        future_prediction = model.predict(last_60_days)
        future_prediction = scaler.inverse_transform(future_prediction)

        # Display the predicted future price
        st.success(f"Predicted Price for {future_date}: ${future_prediction[0][0]:.2f}")

        # Plot historical and predicted prices
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index[-len(predictions):], df['Close'][-len(predictions):], label='Actual Prices')
        ax.plot(df.index[-len(predictions):], predictions, label='Predicted Prices')
        ax.axvline(x=end_date, color='r', linestyle='--', label='Prediction Start')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'{stock} Stock Price Prediction')
        ax.legend()
        st.pyplot(fig)