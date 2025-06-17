from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import pandas as pd
import requests
import numpy as np
from models.regressor import SimpleRegressor
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# --- Global variable to hold our trained model ---
model = None
live_data_df = None
last_location = {'lat': 41.15, 'lng': -8.61}

# Prometheus metrics
prediction_requests_total = Counter('prediction_requests_total', 'Total number of prediction requests')
prediction_errors_total = Counter('prediction_errors_total', 'Total number of prediction errors')
prediction_latency_seconds = Histogram('prediction_latency_seconds', 'Prediction request latency in seconds')

def fetch_live_weather_data(lat=41.15, lng=-8.61):
    """Fetches the last 24 hours of weather data from Open-Meteo for a given location."""
    api_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&hourly=temperature_2m,relativehumidity_2m,apparent_temperature,windspeed_10m&forecast_days=1"
    response = requests.get(api_url)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data['hourly'])
    df.rename(columns={
        'temperature_2m': 'temperature',
        'relativehumidity_2m': 'humidity',
        'apparent_temperature': 'apparent_temperature',
        'windspeed_10m': 'wind_speed'
    }, inplace=True)
    print(f"Successfully fetched live weather data for lat={lat}, lng={lng}.")
    return df.dropna()

def train_model_for_location(lat, lng):
    global model, live_data_df
    print(f"--- Training model for location: lat={lat}, lng={lng} ---")
    live_data_df = fetch_live_weather_data(lat, lng)
    features = live_data_df[['temperature', 'humidity', 'wind_speed']].values
    target = live_data_df['apparent_temperature'].values
    X_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(target, dtype=torch.float32).view(-1, 1)
    input_size = 3
    model = SimpleRegressor(input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 100
    for epoch in range(num_epochs):
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    print("--- Model training complete for location. ---")

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    lat = float(data.get('lat', 41.15))
    lng = float(data.get('lng', -8.61))
    global last_location
    last_location = {'lat': lat, 'lng': lng}
    try:
        train_model_for_location(lat, lng)
        table_html = live_data_df.to_html(classes='data', header="true", index=False)
        return jsonify({'message': f'Model trained for location ({lat:.4f}, {lng:.4f})', 'live_data_table': table_html})
    except Exception as e:
        return jsonify({'message': f'Error: {e}', 'live_data_table': ''}), 500

@app.route('/predict', methods=['POST'])
@prediction_latency_seconds.time()
def predict():
    prediction_requests_total.inc()
    data = request.get_json()
    lat = float(data.get('lat', last_location['lat']))
    lng = float(data.get('lng', last_location['lng']))
    try:
        # Fetch the latest weather data for the location
        df = fetch_live_weather_data(lat, lng)
        # Use the most recent hour's data for prediction
        latest = df.iloc[-1]
        input_features = torch.tensor([[latest['temperature'], latest['humidity'], latest['wind_speed']]], dtype=torch.float32)
        with torch.no_grad():
            predicted_apparent_temp = model(input_features).item()
        prediction = f"Predicted Apparent Temperature: {predicted_apparent_temp:.2f} °C (at {lat:.4f}, {lng:.4f})"
        return jsonify({'prediction': prediction})
    except Exception as e:
        prediction_errors_total.inc()
        return jsonify({'prediction': f'Error: {e}'}), 500

@app.route('/', methods=['GET'])
def home():
    return render_template(
        'index.html',
        live_data_table=live_data_df.to_html(classes='data', header="true", index=False),
        prediction_text=""
    )

# on startup: train once
with app.app_context():
    train_model_for_location(41.15, -8.61)