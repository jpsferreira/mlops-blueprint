# serve-and-monitor-ml

> **Disclaimer:** This project is intended as a demonstration of serving and monitoring a machine learning workflow. The model used is a simple placeholder and is not intended for real predictive accuracy or production use. The aim is to illustrate application structure, monitoring, and UI—not to train a high-quality model.

A Flask-based machine learning web application for predicting the 'feeling' (apparent) temperature using live weather data. The app allows users to select a location on a map, train a regression model for that location, and predict the apparent temperature. The project includes operational monitoring using Prometheus.

## Features
- Select location on an interactive map
- Train a regression model using live weather data for the selected location
- Predict the apparent temperature for the latest weather data
- View the live data used for training in a scrollable table
- Application and prediction monitoring with Prometheus metrics

## Project Structure
```
serve-and-monitor-ml/
├── app/
│   ├── app.py                # Flask application
│   ├── static/
│   │   └── style.css         # Custom styles
│   └── templates/
│       └── index.html        # Main UI template
├── models/
│   └── regressor.py          # PyTorch regression model
├── requirements.txt          # Python dependencies
├── Dockerfile                # Flask app Dockerfile
├── Dockerfile.prometheus     # Prometheus Dockerfile
├── docker-compose.yml        # Multi-service setup
├── prometheus.yml            # Prometheus config
└── README.md                 # This file
```

## Quick Start

### 1. Clone the repository
```sh
git clone <repo-url>
cd serve-and-monitor-ml
```

### 2. Build and run with Docker Compose
```sh
docker compose up --build
```
- The Flask app will be available at [http://localhost:5001](http://localhost:5001)
- Prometheus will be available at [http://localhost:9090](http://localhost:9090)

### 3. Using the App
- Select a location on the map (drag the marker or click).
- Click **Train Model for Location** to fetch live weather data and train the model.
- Click **Predict Feeling Temperature** to get a prediction for the selected location.
- The right-side table shows the live data used for training (scrollable).

## Monitoring with Prometheus

### Metrics Exposed
The Flask app exposes Prometheus metrics at `/metrics` (automatically scraped by Prometheus):
- `prediction_requests_total`: Total number of prediction requests
- `prediction_errors_total`: Total number of prediction errors
- `prediction_latency_seconds`: Histogram of prediction request latency
- Default Flask and HTTP metrics (request count, error rate, latency, etc.)

### Prometheus Setup
- The `prometheus` service is included in `docker-compose.yml` and uses `prometheus.yml` for configuration.
- Prometheus scrapes the Flask app at `web:5001/metrics` every 15 seconds.
- Access Prometheus UI at [http://localhost:9090](http://localhost:9090) to query and visualize metrics.

### Example Prometheus Queries
- Total prediction requests: `prediction_requests_total`
- Prediction error rate: `rate(prediction_errors_total[5m])`
- Average prediction latency: `rate(prediction_latency_seconds_sum[5m]) / rate(prediction_latency_seconds_count[5m])`

## Development

### Install dependencies (if running locally)
```sh
pip install -r requirements.txt
```

### Export requirements from Poetry (if using Poetry)
```sh
poetry export -f requirements.txt --without-hashes > requirements.txt
```

### Run Flask app locally
```sh
export FLASK_APP=app/app.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5001
```

## Notes
- The app uses live weather data from Open-Meteo API.
- Model is retrained each time you select a new location and click train.
- Prometheus metrics are available at `/metrics` on the Flask app.
- For advanced dashboards, consider adding Grafana (not included by default).

---

Feel free to open issues or contribute!