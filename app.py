from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load the trained model from the pickle file
with open('/home/shahrin/mysite/rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load test data
test_data = pd.read_csv('/home/shahrin/mysite/test.csv')

# Define the price range mapping
price_range_map = {0: 'low cost', 1: 'medium cost', 2: 'high cost', 3: 'very high cost'}


@app.route('/api/predict', methods=['POST'])
def predict_price():
    # Get device specifications from request JSON
    data = request.json
    battery_power = data.get('battery_power')
    ram = data.get('ram')
    px_height = data.get('px_height')
    px_width = data.get('px_width')

    # Predict price using the loaded model
    predicted_price = model.predict([[battery_power, ram, px_height * px_width]])[0]

    # Map predicted price to price range
    predicted_price_range = price_range_map[predicted_price]

    return jsonify({'predicted_price_range': predicted_price_range})


@app.route('/api/devices', methods=['POST'])
def get_all_devices():
    # Retrieve a list of all devices from the test data
    all_devices = test_data.to_dict(orient='records')
    return jsonify(all_devices)


@app.route('/api/devices/<int:id>', methods=['GET'])
def get_device_details(id):
    # Retrieve details of a specific device by ID
    device_details = test_data.iloc[id].to_dict()
    return jsonify(device_details)


@app.route('/api/devices', methods=['POST'])
def add_new_device():
    # Add a new device to the test data
    new_device = request.json
    test_data.append(new_device, ignore_index=True)
    return jsonify({'message': 'New device added successfully'})


@app.route('/api/predict/<int:device_id>', methods=['POST'])
def predict_and_save_price(device_id):
    # Predict price for the specified device
    device_specifications = test_data.iloc[device_id][['battery_power', 'ram', 'px_height', 'px_width']]
    predicted_price = model.predict([device_specifications])[0]

    # Save the predicted price range in the device entity
    test_data.at[device_id, 'predicted_price_range'] = price_range_map[predicted_price]
    return jsonify({'message': 'Predicted price range saved successfully'})


if __name__ == '__main__':
    app.run(debug=True)
