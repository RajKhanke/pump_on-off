from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import pandas as pd
import io
import requests
import threading
import time
from PIL import Image  # Import for image processing

app = Flask(__name__)

# Load models
pump_model = joblib.load('pump_status_dt_model.pkl')
soil_model = load_model('soil_classification_model.h5')

# Dictionaries for crop types, regions, etc.
crop_types = {'BANANA': 0, 'BEAN': 1, 'CABBAGE': 2, 'CITRUS': 3, 'COTTON': 4,
              'MAIZE': 5, 'MELON': 6, 'MUSTARD': 7, 'ONION': 8, 'OTHER': 9,
              'POTATO': 10, 'RICE': 11, 'SOYABEAN': 12, 'SUGARCANE': 13,
              'TOMATO': 14, 'WHEAT': 15}

soil_types = {'DRY': 0, 'HUMID': 1, 'WET': 2}
regions = {'DESERT': 0, 'HUMID': 1, 'SEMI ARID': 2, 'SEMI HUMID': 3}
weather_conditions = {'SUNNY': 0, 'RAINY': 1, 'WINDY': 2, 'NORMAL': 3}
irrigation_types = {'Drip Irrigation': 0, 'Manual Irrigation': 1,
                    'Sprinkler Irrigation': 2, 'Subsurface Irrigation': 3,
                    'Surface Irrigation': 4}

soil_labels = {1: 'Black Soil', 2: 'Clay Soil', 0: 'Alluvial Soil', 3: 'Red Soil'}

# Global variables
soil_moisture_data = []
pump_status = "Off"
previous_pump_status = "Off"
graph_data = []


# Function to fetch weather data
def get_weather(city):
    api_key = "b3c62ae7f7ad5fc3cb0a7b56cb7cbda6"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        temp = data['main']['temp']
        pressure = data['main']['pressure']
        humidity = data['main']['humidity']
        weather_desc = data['weather'][0]['main']
        return temp, pressure, humidity, weather_desc
    except requests.exceptions.HTTPError:
        return None, None, None, None


# Function to map soil type to pump model's expected format
def map_soil_to_pump_model(soil_label):
    if soil_label in ['Black Soil', 'Red Soil']:
        return 'DRY'
    elif soil_label == 'Clay Soil':
        return 'WET'
    elif soil_label == 'Alluvial Soil':
        return 'HUMID'
    return None


# Function to run predictions for all soil moisture values
# Function to run predictions for all soil moisture values
def run_predictions(crop_type, soil_type_for_pump, region, temperature, pressure, humidity, crop_age, irrigation_type, auto_weather_condition):
    global pump_status, graph_data, previous_pump_status
    pump_status = "Off"
    previous_pump_status = "Off"
    graph_data = []

    for soil_moisture in soil_moisture_data:
        try:
            soil_moisture_value = float(soil_moisture)  # Ensure this is a float
        except ValueError:
            print(f"Skipping invalid soil moisture value: {soil_moisture}")
            continue

        # Prepare features for pump prediction
        features = np.array([crop_types[crop_type], soil_types[soil_type_for_pump],
                             regions[region], temperature if temperature else 0,
                             weather_conditions.get(auto_weather_condition, 0),
                             pressure if pressure else 0, humidity if humidity else 0,
                             int(crop_age), irrigation_types[irrigation_type],
                             soil_moisture_value]).reshape(1, -1)

        # Make the pump prediction
        pump_prediction = pump_model.predict(features)
        pump_status = 'On' if pump_prediction[0] == 1 else 'Off'
        graph_data.append((soil_moisture_value, 1 if pump_status == 'On' else -1))  # Update status to -1 for Off

        print(f"Predicted Pump Status: {pump_status} for Soil Moisture: {soil_moisture_value}")  # Debugging output

        # Play sound if pump is Off and it wasn't Off previously
        if pump_status == "Off" and previous_pump_status != "Off":
            play_sound()

        previous_pump_status = pump_status

        # Wait for 1 second before next prediction
        time.sleep(2)


def play_sound():
    # You can use any sound file here
    print("Beep! Pump is Off.")  # Placeholder for actual sound functionality


# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    global soil_moisture_data

    city = crop_type = region = crop_age = irrigation_type = None
    temperature = pressure = humidity = weather_desc = auto_weather_condition = None
    soil_image_url = None

    if request.method == 'POST':
        city = request.form.get('city', '')
        crop_type = request.form.get('crop_type', '')
        region = request.form.get('region', '')
        crop_age = request.form.get('crop_age', '')
        irrigation_type = request.form.get('irrigation_type', '')

        # Handle CSV file upload
        if 'soil_moisture' in request.files:
            soil_moisture_file = request.files['soil_moisture']
            if soil_moisture_file:
                # Read CSV file
                df = pd.read_csv(soil_moisture_file)
                soil_moisture_data = df['Soil Moisture'].tolist()

        # Handle soil image upload
        soil_image_file = request.files.get('soil_image')
        if soil_image_file:
            # Load and preprocess the image for prediction
            image = Image.open(io.BytesIO(soil_image_file.read()))
            image = image.resize((150, 150))
            image = np.array(image) / 255.0
            if image.shape[-1] == 4:
                image = image[..., :3]
            image = np.expand_dims(image, axis=0)

            # Predict the soil type
            soil_pred = soil_model.predict(image)
            soil_label = soil_labels[np.argmax(soil_pred)]
            soil_type_for_pump = map_soil_to_pump_model(soil_label)
        else:
            soil_type_for_pump = request.form.get('soil_type')

        if city:
            temperature, pressure, humidity, weather_desc = get_weather(city)
            auto_weather_condition = "NORMAL"  # Default weather condition
            if weather_desc:
                if 'sunny' in weather_desc.lower():
                    auto_weather_condition = 'SUNNY'
                elif 'rain' in weather_desc.lower():
                    auto_weather_condition = 'RAINY'
                elif 'wind' in weather_desc.lower():
                    auto_weather_condition = 'WINDY'

        if 'predict' in request.form:
            # Start a thread for predictions
            threading.Thread(target=run_predictions, args=(
                crop_type, soil_type_for_pump, region, temperature, pressure, humidity, crop_age, irrigation_type, auto_weather_condition)).start()
            return redirect(url_for('predict'))

    return render_template('index.html', temperature=temperature, pressure=pressure,
                           humidity=humidity, weather_desc=weather_desc, crop_types=crop_types,
                           regions=regions, irrigation_types=irrigation_types, soil_types=soil_types,
                           crop_type=crop_type, region=region, crop_age=crop_age,
                           irrigation_type=irrigation_type, city=city, soil_image_url=soil_image_url)


# Prediction route
@app.route('/predict', methods=['GET'])
def predict():
    global pump_status, graph_data
    return render_template('predict.html', pump_status=pump_status, graph_data=graph_data)


# Update graph data every second
@app.route('/update_graph', methods=['GET'])
def update_graph():
    global graph_data
    return jsonify(graph_data)


# Update pump status every second
@app.route('/update_pump_status', methods=['GET'])
def update_pump_status():
    global pump_status
    return jsonify({'pump_status': pump_status})


if __name__ == '__main__':
    app.run(debug=True)
