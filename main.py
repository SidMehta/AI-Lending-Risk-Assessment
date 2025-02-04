# Standard library imports
import os
import json
import pickle
import sys
import logging
import traceback
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import flask
from flask import Flask, request, render_template, jsonify
from google.cloud import aiplatform
from google.cloud import logging as cloud_logging
from google.cloud.logging.handlers import CloudLoggingHandler

# Create Flask app
app = Flask(__name__)
import logging
import traceback
import sys
from datetime import datetime

app = Flask(__name__)

# Setup Google Cloud Logging
client = cloud_logging.Client()
handler = CloudLoggingHandler(client)
cloud_logger = logging.getLogger('cloudLogger')
cloud_logger.setLevel(logging.INFO)
cloud_logger.addHandler(handler)

# Also log to stdout for App Engine logging
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
cloud_logger.addHandler(stream_handler)

def log_error(error, context=""):
    """Enhanced error logging function"""
    error_details = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'timestamp': datetime.now().isoformat(),
        'context': context,
        'traceback': traceback.format_exc()
    }
    cloud_logger.error(f"Error occurred: {json.dumps(error_details)}")
    return error_details

# Load preprocessing artifacts
try:
    cloud_logger.info("Starting application initialization...")
    
    ARTIFACTS_FOLDER = os.path.join(os.path.dirname(__file__), 'artifacts')
    cloud_logger.info(f"Artifacts folder path: {ARTIFACTS_FOLDER}")
    cloud_logger.info(f"Current directory contents: {os.listdir()}")
    
    with open(os.path.join(ARTIFACTS_FOLDER, 'selected_features.json'), 'r') as f:
        SELECTED_FEATURES = json.load(f)
    cloud_logger.info(f"Loaded selected features: {len(SELECTED_FEATURES)}")
    
    with open(os.path.join(ARTIFACTS_FOLDER, 'numeric_features.json'), 'r') as f:
        NUMERIC_FEATURES = json.load(f)
    cloud_logger.info(f"Loaded numeric features: {len(NUMERIC_FEATURES)}")
    
    with open(os.path.join(ARTIFACTS_FOLDER, 'scaler.pkl'), 'rb') as f:
        SCALER = pickle.load(f)
    cloud_logger.info("Loaded scaler successfully")
    
except Exception as e:
    error_details = log_error(e, "Failed to load preprocessing artifacts")
    raise RuntimeError(f"Application startup failed: {error_details}")

# Initialize Vertex AI
try:
    PROJECT_ID = os.getenv('PROJECT_ID')
    LOCATION = os.getenv('LOCATION', 'us-central1')
    ENDPOINT_ID = os.getenv('ENDPOINT_ID')
    
    if not all([PROJECT_ID, ENDPOINT_ID]):
        raise ValueError("Missing required environment variables: PROJECT_ID or ENDPOINT_ID")
    
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    endpoint = aiplatform.Endpoint(
        endpoint_name=f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"
    )
    cloud_logger.info(f"Successfully initialized Vertex AI with project {PROJECT_ID}")
    
except Exception as e:
    error_details = log_error(e, "Failed to initialize Vertex AI")
    raise RuntimeError(f"Application startup failed: {error_details}")

# Default values with current date
CURRENT_YEAR = datetime.now().year
CURRENT_QUARTER = ((datetime.now().month - 1) // 3) + 1
CURRENT_MONTH = datetime.now().month

# Input validation configurations
INPUT_VALIDATIONS = {
    'orig_amt': {'min': 10000, 'max': 2000000, 'type': float},
    'cscore_b': {'min': 300, 'max': 850, 'type': int},
    'oltv': {'min': 0, 'max': 100, 'type': float},
    'dti': {'min': 0, 'max': 65, 'type': float},
    'orig_rt': {'min': 0, 'max': 25, 'type': float}
}

VALID_STATES = ['AL', 'AR', 'CA', 'CO', 'FL', 'GA', 'IL', 'LA', 'MA', 'MD', 
                'MI', 'MN', 'MS', 'NC', 'NH', 'NJ', 'OH', 'OK', 'OR', 'PA', 
                'SC', 'TN', 'TX', 'VA', 'WA', 'WI', 'WV', 'NV', 'NY']

# Initialize default values
DEFAULT_VALUES = {
    'Year_orig': CURRENT_YEAR,
    'Quarter_orig': CURRENT_QUARTER,
    'orig_rt': 3.5,
    'orig_amt': 200000,
    'oltv': 80.0,
    'ocltv': 85.0,
    'dti': 35.0,
    'cscore_b': 750,
    'mi_pct': 0.0,
    'num_bo': 1,
    'num_unit': 1,
    'year_x': CURRENT_YEAR,
    'year_y': CURRENT_YEAR,
    'quarter': CURRENT_QUARTER,
    'HPI_state': 200,
    'month': CURRENT_MONTH,
    'FRM30_rate': 3.0,
    'treasury_3mon_rate': 0.5,
    'source_FN': 0,
    'fthb_flg_Y': 0,
}

# Initialize state columns
for state in VALID_STATES:
    DEFAULT_VALUES[f'state_{state}'] = 0

def validate_and_transform_input(data):
    """Validate and transform input data"""
    transformed_data = {}
    errors = []
    
    for field, validation in INPUT_VALIDATIONS.items():
        try:
            if field not in data:
                errors.append(f"Missing required field: {field}")
                continue
                
            value = validation['type'](data[field])
            if not validation['min'] <= value <= validation['max']:
                errors.append(f"{field} must be between {validation['min']} and {validation['max']}")
            transformed_data[field] = value
            
        except (ValueError, TypeError):
            errors.append(f"Invalid value for {field}: must be a {validation['type'].__name__}")
    
    if 'state' not in data:
        errors.append("Missing required field: state")
    elif not isinstance(data['state'], str) or data['state'] not in VALID_STATES:
        errors.append(f"Invalid state code. Must be one of: {', '.join(VALID_STATES)}")
    else:
        transformed_data['state'] = data['state']
    
    if errors:
        raise ValueError('\n'.join(errors))
    
    return transformed_data

def prepare_model_input(validated_data):
    """Prepare input data for model prediction"""
    try:
        # Start with default values
        model_input = DEFAULT_VALUES.copy()
        
        # Update with validated user input
        for key, value in validated_data.items():
            if key != 'state':
                model_input[key] = value
        
        # Handle state encoding
        state = validated_data['state']
        # Reset all state columns
        for s in VALID_STATES:
            model_input[f'state_{s}'] = 0
        # Set selected state
        model_input[f'state_{state}'] = 1
        
        # Create DataFrame
        input_df = pd.DataFrame([model_input])
        
        # Scale numeric features
        numeric_cols = [col for col in NUMERIC_FEATURES if col in input_df.columns]
        if numeric_cols:
            input_df[numeric_cols] = SCALER.transform(input_df[numeric_cols])
        
        # Select features in correct order
        final_input = input_df[SELECTED_FEATURES]
        
        # Validate final input
        if final_input.isnull().any().any():
            raise ValueError("Final input contains null values")
            
        cloud_logger.info(f"Prepared model input with shape {final_input.shape}")
        return final_input
        
    except Exception as e:
        error_details = log_error(e, "Error in prepare_model_input")
        raise ValueError(f"Failed to prepare model input: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    cloud_logger.info(f"Received prediction request {request_id}")
    
    try:
        # Get and validate input data
        if not request.is_json:
            raise ValueError("Request must be JSON")
        
        data = request.json
        if not data:
            raise ValueError("No input data provided")
        
        cloud_logger.info(f"Request {request_id} data: {json.dumps(data)}")
        
        # Validate and transform input
        validated_data = validate_and_transform_input(data)
        cloud_logger.info(f"Validated data: {validated_data}")
        
        # Prepare model input
        model_input = prepare_model_input(validated_data)
        cloud_logger.info(f"Model input shape: {model_input.shape}")
        
        # Make prediction
        instances = model_input.values.tolist()
        prediction_response = endpoint.predict(instances=instances)
        cloud_logger.info(f"Raw prediction response: {prediction_response}")
        
        # Extract prediction value with better error handling
        try:
            predictions = prediction_response.predictions[0]
            # Handle both list and single value formats
            prediction_value = float(predictions[0] if isinstance(predictions, (list, tuple)) else predictions)
            cloud_logger.info(f"Raw predictions: {predictions}")
            cloud_logger.info(f"Extracted prediction value: {prediction_value}")
        except Exception as e:
            cloud_logger.error(f"Error extracting prediction: {str(e)}")
            cloud_logger.error(f"Raw prediction response: {prediction_response}")
            cloud_logger.error(f"Prediction type: {type(prediction_response.predictions[0])}")
            raise ValueError("Failed to process model prediction")
        
        # Prepare response
        response = {
            'request_id': request_id,
            'prediction': prediction_value,
            'risk_level': 'High' if prediction_value > 0.5 else 'Low',
            'confidence': prediction_value if prediction_value > 0.5 else (1 - prediction_value),
            'timestamp': datetime.now().isoformat()
        }
        
        cloud_logger.info(f"Request {request_id} completed successfully")
        return jsonify(response)
        
    except ValueError as ve:
        error_details = log_error(ve, f"Validation error in request {request_id}")
        return jsonify({
            'error': str(ve),
            'request_id': request_id,
            'type': 'ValidationError'
        }), 400
        
    except Exception as e:
        error_details = log_error(e, f"Unexpected error in request {request_id}")
        return jsonify({
            'error': 'An unexpected error occurred',
            'request_id': request_id,
            'type': 'ServerError'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)