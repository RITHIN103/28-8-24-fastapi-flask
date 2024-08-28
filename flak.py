from flask import Flask, request, render_template, Response, jsonify
from flask_cors import CORS, cross_origin
import os
import json
import logging

from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
from predictFromModel import prediction

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict_route_client():
    try:
        data = request.get_json() if request.is_json else request.form
        path = data.get('filepath')
        if not path:
            return jsonify({"error": "Filepath is required"}), 400
        
        pred_val = pred_validation(path)
        pred_val.prediction_validation()
        
        pred = prediction(path)
        path, json_predictions = pred.predictionFromModel()
        
        return jsonify({
            "message": f"Prediction File created at {path}",
            "predictions": json.loads(json_predictions)
        })
    except (ValueError, KeyError) as e:
        logging.error(f"ValueError or KeyError: {e}")
        return jsonify({"error": f"Error Occurred: {str(e)}"}), 400
    except Exception as e:
        logging.error(f"Exception: {e}")
        return jsonify({"error": f"Error Occurred: {str(e)}"}), 500

@app.route("/train", methods=['POST'])
@cross_origin()
def train_route_client():
    try:
        data = request.get_json()
        path = data.get('folderPath')
        if not path:
            return jsonify({"error": "Folder path is required"}), 400
        
        train_val_obj = train_validation(path)
        train_val_obj.train_validation()
        
        train_model_obj = trainModel()
        train_model_obj.trainingModel()
        
        return jsonify({"message": "Training successful!"})
    except (ValueError, KeyError) as e:
        logging.error(f"ValueError or KeyError: {e}")
        return jsonify({"error": f"Error Occurred: {str(e)}"}), 400
    except Exception as e:
        logging.error(f"Exception: {e}")
        return jsonify({"error": f"Error Occurred: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
