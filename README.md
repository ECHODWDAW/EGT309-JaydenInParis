
# EGT309 - Predictive Maintenance Using Machine Learning

## Project Overview

This project aims to predict whether industrial machines require maintenance using machine sensor readings and machine learning.

The project includes:
- Exploratory Data Analysis (EDA)
- Data cleaning and validation
- Feature engineering
- Data preprocessing
- Training and evaluation of four ML models
- Docker containerisation
- Model comparison and selection

## Dataset

Dataset: `data/proj_machine_readings.csv`

The dataset contains:
- 100,000 records
- 50 machines
- 13 original columns
- Target variable: `maintenance_required`

Sensor features include:
- temperature
- vibration
- humidity
- pressure
- energy_consumption

Target classes:
- 0: No maintenance required
- 1: Maintenance required

The original dataset is preserved without modification.

## Members

- Ethan Kwan Hong Wei
- Loo Yan Yi
- Seah Wei Jie

## Project Structure

```text
EGT309-JaydenInParis/
|-- config/
|   `-- config.yaml
|-- data/
|   `-- proj_machine_readings.csv
|-- src/
|   |-- __init__.py
|   |-- data_cleaning.py
|   |-- preprocessing.py
|   `-- evaluation.py
|-- eda.ipynb
|-- Dockerfile
|-- .dockerignore
|-- .gitignore
|-- requirements.txt
|-- run.sh
`-- README.md
```

## Requirements

- Docker Desktop (or Docker Engine)
- Git
- Web browser

Python dependencies are installed automatically inside Docker.

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/ECHODWDAW/EGT309-JaydenInParis.git
cd EGT309-JaydenInParis
```

Private repository access is required.

### 2. Build the Docker image

```bash
docker build -t egt309-ml:latest .
```

### 3. Start JupyterLab

For Windows PowerShell:

```powershell
docker run --rm -it -p 127.0.0.1:8888:8888 -v "${PWD}:/app" --name egt309-container egt309-ml:latest
```

For Linux/macOS with Bash:

```bash
bash run.sh
```

The script builds the image and starts JupyterLab.

### 4. Open JupyterLab

Copy the JupyterLab URL containing the authentication token from the terminal into your browser.

Open `eda.ipynb` to view the exploratory data analysis.

The project directory is mounted into the container, so changes to files are saved on the host computer.

To stop JupyterLab, press `Ctrl+C` in the terminal.

## Exploratory Data Analysis

The EDA investigates:
- Missing values and duplicates
- Machine maintenance class distribution
- Sensor distributions and outliers
- Sensor relationships with maintenance requirements
- Machine-level patterns
- Potential target leakage

Key findings:
- 14,000 missing temperature readings
- 6,703 negative vibration readings
- No duplicate records
- Approximately 80.3% of records belong to Class 0
- Approximately 19.7% of records belong to Class 1

The dataset exhibits class imbalance.

## Data Cleaning

Implemented in `src/data_cleaning.py` using the `DataCleaner` class.

Cleaning operations include:
- Dataset loading and validation
- Data type conversion
- Invalid record checking
- Duplicate record checking
- Vibration error handling
- Sensor value validation
- Chronological sorting

### Vibration Correction

EDA identified a systematic pattern of negative vibration readings at minutes 00, 15, 30 and 45.

Based on the suspected sign-inversion pattern, 6,667 readings were converted to positive values while preserving their magnitude.

This correction is based on an observed data pattern rather than a confirmed hardware fault.

The remaining 36 negative vibration readings were converted to missing values.

### Cleaning Results

| Data Quality Issue | Before | After |
|---|---:|---:|
| Total records | 100,000 | 100,000 |
| Duplicate records | 0 | 0 |
| Missing temperature | 14,000 | 14,000 |
| Negative vibration | 6,703 | 0 |
| Missing vibration | 0 | 36 |

Missing sensor values are handled during preprocessing.

## Feature Engineering

Implemented in `src/preprocessing.py` using the `FeatureEngineer` class.

Engineered features include:
- hour_sin
- hour_cos
- day_sin
- day_cos
- temperature_missing
- vibration_missing

Cyclical time features represent periodic patterns in hours and days.

Missing-value indicators preserve information about unavailable sensor readings.

### Potential Data Leakage

The following derived columns are excluded from model inputs as a precaution against potential target leakage:

- machine_status
- anomaly_flag
- predicted_remaining_life
- failure_type
- downtime_risk

The target variable and original timestamp are also excluded from the final model input features.

## Data Preprocessing

Implemented in `src/preprocessing.py` using the `DataPreprocessor` class.

Preprocessing operations include:
- Chronological dataset splitting
- Median imputation
- Standard scaling
- One-hot encoding of machine IDs
- Feature transformation

Imputation, scaling and encoding are fitted using the training dataset only to prevent test-set information from influencing preprocessing.

### Dataset Splitting

The dataset is divided chronologically using an 80:20 training and testing split.

Split timestamp: `2025-02-25 13:20:00`

| Dataset | Records | Features |
|---|---:|---:|
| Training | 80,000 | 61 |
| Testing | 20,000 | 61 |

### Target Distribution

| Class | Training | Testing |
|---|---:|---:|
| No maintenance required | 64,284 | 16,019 |
| Maintenance required | 15,716 | 3,981 |

### Final Feature Composition

| Feature Category | Count |
|---|---:|
| Original sensor features | 5 |
| Cyclical time features | 4 |
| One-hot encoded machine IDs | 50 |
| Missing-value indicators | 2 |
| Total | 61 |

No missing values remain in the transformed training and testing datasets.

## Shared Pipeline Usage

All models should use the same dataset splitting and preprocessing approach to ensure fair comparison.

### Load the processed dataset

```python
from src.preprocessing import DataPreprocessor

processor = DataPreprocessor()

X_train, X_test, y_train, y_test = processor.run()
```

The returned datasets can be used for model training and final evaluation.

For cross-validation and hyperparameter tuning, use the raw split and fit preprocessing separately within each training fold.

```python
processor = DataPreprocessor()

X_train, X_test, y_train, y_test = processor.prepare_raw_data()

preprocessor = processor.build_preprocessor()
```

The preprocessor can then be included in a scikit-learn Pipeline.

Time-aware validation should be used, keeping identical timestamps together where applicable.

The final testing dataset should not be used for hyperparameter tuning.

## Machine Learning Models

Planned models:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Histogram-based Gradient Boosting

## Model Evaluation

Implemented in `src/evaluation.py` using the `ModelEvaluator` class.

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision
- Classification report
- Confusion matrix

The evaluation module also provides functions for:
- Confusion matrix visualisation
- ROC curve visualisation
- Precision-recall curve visualisation
- CSV export of evaluation metrics

### Evaluation Usage

After training a model:

```python
from src.evaluation import ModelEvaluator

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

evaluator = ModelEvaluator("Model Name")

results = evaluator.run(y_test, y_pred, y_prob)
```

The probability input represents the probability of Class 1.

### Baseline Model

A DummyClassifier using the most frequent class strategy was used for integration testing and baseline comparison.

| Metric | Result |
|---|---:|
| Accuracy | 80.10% |
| Precision | 0.00% |
| Recall | 0.00% |
| F1-score | 0.00% |
| ROC-AUC | 0.5000 |
| Average Precision | 0.1991 |

The baseline classifier predicts every record as not requiring maintenance.

Despite achieving approximately 80% accuracy, it fails to identify all 3,981 maintenance cases in the testing dataset.

This demonstrates why accuracy alone is insufficient for evaluating an imbalanced predictive maintenance dataset.

The baseline is not one of the four final machine learning models.

## Configuration

Shared project settings are stored in:

`config/config.yaml`

Configurable settings include:
- Dataset path
- Target variable
- Timestamp and machine ID columns
- Training and testing split
- Data cleaning options
- Vibration correction settings
- Imputation strategy
- Sensor feature selection
- Excluded features

## Pipeline Testing

With the Docker container running, execute the following commands from a separate terminal.

### Test data cleaning

```bash
docker exec egt309-container python -m src.data_cleaning
```

### Test preprocessing

```bash
docker exec egt309-container python -m src.preprocessing
```

### Test model evaluation

```bash
docker exec egt309-container python -m src.evaluation
```

Replace `egt309-container` with the actual running container name if different.

Data cleaning, preprocessing and the main evaluation functions have been successfully integrated and tested inside Docker using the baseline classifier.

## Project Status

- [x] Initial EDA
- [x] Docker environment setup
- [x] GitHub repository setup
- [x] Data cleaning and validation
- [x] Feature engineering
- [x] Shared dataset splitting
- [x] Data preprocessing
- [x] Shared model evaluation implementation
- [x] Baseline model integration testing
- [ ] Model training and hyperparameter tuning
- [ ] Model comparison and evaluation
- [ ] Final model selection
- [ ] Final documentation and submission
