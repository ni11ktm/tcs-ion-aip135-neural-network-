# TCS iON AIP 135 — Neural Network: Customer Buying Trends Prediction

## Project Overview
This project predicts whether a customer will buy a product based on demographic and behavioral signals. It solves the problem of identifying likely buyers early so marketing and personalization workflows can prioritize high-intent users. The deployed model pipeline uses a Perceptron architecture with feature scaling and API-ready preprocessing for categorical and numeric inputs. The achieved project accuracy is 99.51%.

## Accuracy Results
| Metric     | Value  |
|------------|--------|
| Accuracy   | 99.51% |
| Precision  | 99.48% |
| Recall     | 99.53% |
| F1-Score   | 99.50% |

## Project Structure
```text
tcs-ion-aip135-neural-network/
├── data/
│   └── customer_data_sample.csv       # Small demo dataset (<=100 rows)
├── notebooks/
│   ├── milestone1_eda_preprocessing.ipynb          # Milestone 1 notebook
│   ├── milestone2_model_training_evaluation.ipynb  # Milestone 2 notebook
│   └── milestone3_finetuning_deployment.ipynb      # Milestone 3 notebook
├── src/
│   ├── __init__.py                    # Package marker
│   ├── preprocess.py                  # Cleaning and feature engineering utilities
│   ├── train.py                       # Train/evaluate script
│   ├── evaluate.py                    # Metrics and visualization generation
│   ├── predict.py                     # Standalone single-record prediction
│   └── save_model.py                  # Final model + metadata export
├── api/
│   ├── __init__.py                    # Package marker
│   └── app.py                         # Flask REST API service
├── models/
│   └── .gitkeep                       # Placeholder; binary models are gitignored
├── tests/
│   └── test_api.py                    # API smoke and validation tests
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Production dependencies
├── requirements-dev.txt               # Development/test dependencies
└── README.md                          # Project documentation
```

## Setup and Installation
1. Clone: `git clone https://github.com/ni11ktm/tcs-ion-aip135-neural-network`
2. Create virtual environment: `python -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Train the model: `python src/save_model.py`
5. Start the API: `python api/app.py`

## API Usage

### Health check
```bash
curl http://localhost:5000/health
```
Example response:
```json
{"status":"ok","model":"Perceptron","accuracy":99.51}
```

### Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":34,"gender":"Female","annual_income":72000,"purchase_history":14,"product_category":"Electronics","loyalty_score":7.8,"time_on_site":18.5}'
```
Example response:
```json
{
  "prediction": 1,
  "label": "Will Buy",
  "probability": 0.97,
  "model_version": "1.0.0",
  "input_received": {
    "age": 34,
    "gender": "Female",
    "annual_income": 72000,
    "purchase_history": 14,
    "product_category": "Electronics",
    "loyalty_score": 7.8,
    "time_on_site": 18.5
  }
}
```

### Model info
```bash
curl http://localhost:5000/model-info
```
Example response:
```json
{
  "model_type": "Perceptron",
  "accuracy_on_test": 99.51,
  "f1_score": 99.50,
  "model_version": "1.0.0"
}
```

## Project Milestones
| Milestone | Deliverables | Status |
|-----------|--------------|--------|
| Milestone 1 | EDA, data cleaning, baseline preprocessing | Complete |
| Milestone 2 | Model training, evaluation, and metrics analysis | Complete |
| Milestone 3 | Fine-tuning, model export, deployment-ready API | Complete |

## Technologies Used
- Python: core implementation language.
- Scikit-learn: model pipeline, training, evaluation.
- Pandas and NumPy: tabular processing and numeric transformations.
- Matplotlib and Seaborn: visual analysis artifacts.
- Flask and Flask-CORS: REST API service and CORS handling.
- Joblib: model serialization for deployment.
- Pytest: API endpoint testing.
- Jupyter Notebook: milestone documentation and experiments.
- Git and GitHub: version control and project hosting.

## Author
ni11ktm — TCS iON AIP 135 Industry Project, April 2026
