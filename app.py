from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from POST request
        data = request.get_json(force=True)

        # Convert all values to float automatically
        features = [float(value) for value in data.values()]

        # Convert to numpy array and reshape for single prediction
        final_features = np.array(features).reshape(1, -1)

        # Predict
        prediction = model.predict(final_features)

        # Return result
        return jsonify({'PredictedPrice': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
