import joblib 
import numpy as np

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict_get():
    # Assuming you want to use this data when a request is made to /predict
    X_text = np.array([7.594444821, 7.479555538, 1.616463184, 1.53352356, 0.796666503, 0.635422587, 0.362012237, 0.315963835, 2.277026653])

    # Assuming your model takes X_text as input and returns predictions
    predictions = model.predict(X_text.reshape(1, -1))

    # Convert predictions to JSON and return
    return jsonify(predictions.tolist())


@app.route('/pso', methods=['POST'])
def predict_pos():
    try:
        # Get the data from the request
        data = request.get_json(force=True)
        
        # Assuming the data in the request is in the same format as X_text
        X_text = np.array(data['input_data'])

        # Assuming your model takes X_text as input and returns predictions
        predictions = model.predict(X_text.reshape(1, -1))

        # Convert predictions to JSON and return
        return jsonify(predictions.tolist())

    except Exception as e:
        # Handle errors appropriately (e.g., invalid input format, model errors)
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    model = joblib.load('models/best_model_0.01.pkl')
    app.run(port=8080)
