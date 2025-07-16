import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def prepare_data(data, sequence_length=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def forecast_next_7_days(data):
    data = data[['Close']].dropna()
    X, y, scaler = prepare_data(data)

    model = build_model((X.shape[1], 1))
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    # ✅ RMSE Calculation (on training predictions)
    y_pred_train = model.predict(X, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y, y_pred_train))

    last_sequence = X[-1]
    predictions = []

    for _ in range(7):
        pred = model.predict(last_sequence.reshape(1, X.shape[1], 1), verbose=0)
        predictions.append(pred[0, 0])

        # Update sequence
        last_sequence = np.append(last_sequence[1:], [[pred[0, 0]]], axis=0)

    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return forecast , rmse
