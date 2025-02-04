<img width="934" alt="image" src="https://github.com/user-attachments/assets/e5ce5998-a351-42c5-84bc-6d23f0757c2c" />


# Loan Default Risk Assessment Application

This is a web application deployed on Google App Engine that provides real-time loan default risk assessment using a machine learning model deployed on Vertex AI. The application takes loan details as input and returns the probability of default along with risk assessment metrics.

## Features

- Real-time loan default risk prediction
- Web-based interface for loan officers
- Integration with Vertex AI for model inference
- Comprehensive input validation
- Detailed risk assessment results including:
  - Default probability
  - Risk level classification
  - Confidence score
  - Request tracking

## Prerequisites

- Google Cloud Platform account
- Python 3.9
- Google Cloud SDK
- A trained model deployed on Vertex AI

## Project Structure
```
lendingapp/
├── main.py               # Main application file
├── app.yaml             # App Engine configuration
├── requirements.txt     # Python dependencies
├── artifacts/
│   ├── numeric_features    # List of numeric features
│   ├── selected_features   # List of all features used by model
│   └── scaler.pkl         # Fitted StandardScaler object
└── templates/
    └── index.html         # Frontend interface
```

## Configuration

1. Update `app.yaml` with your project details:
```yaml
env_variables:
  PROJECT_ID: "your-project-id"
  LOCATION: "us-central1"
  ENDPOINT_ID: "your-endpoint-id"
  GOOGLE_CLOUD_PROJECT: "your-project-id"
```

2. Ensure your Vertex AI model endpoint is deployed and accessible.

## Input Features

The application accepts the following inputs:
- Original Loan Amount ($): 10,000 - 2,000,000
- Credit Score: 300 - 850
- Original Loan-to-Value Ratio (%): 0 - 100
- Debt-to-Income Ratio (%): 0 - 65
- Original Interest Rate (%): 0 - 25
- State: Valid US state codes

## Deployment

1. Clone the repository:
```bash
git clone [repository-url]
cd lendingapp
```

2. Install dependencies locally (for testing):
```bash
pip install -r requirements.txt
```

3. Deploy to App Engine:
```bash
gcloud app deploy app.yaml
```

4. View the deployed application:
```bash
gcloud app browse
```

## Understanding Results

The application provides:
- **Default Probability**: The model's predicted probability of loan default
- **Risk Level**: Classification as "High" (>50%) or "Low" (≤50%)
- **Confidence**: The model's confidence in its prediction
- **Request ID**: Unique identifier for tracking predictions
- **Timestamp**: When the prediction was made

## Error Handling

The application includes:
- Input validation for all fields
- Proper error messages for invalid inputs
- Logging for debugging and monitoring
- Error tracking with request IDs

## Monitoring

Monitor the application using Google Cloud Console:
- App Engine dashboard for application metrics
- Cloud Logging for detailed logs
- Vertex AI dashboard for model metrics

## Security

- HTTPS enforced for all endpoints
- Input validation to prevent injection attacks
- Service account authentication for Vertex AI
- Proper error handling to prevent information leakage

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Acknowledgments

- Based on the Vertex AI platform
- Uses Flask framework for web application
- Deployed on Google App Engine
