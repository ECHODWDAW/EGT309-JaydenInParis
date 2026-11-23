
# EGT309 — Predictive Maintenance Using Machine Learning

## Project Overview

This project aims to predict whether industrial machines require maintenance using machine sensor readings and machine learning.

The project includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Training and evaluation of three ML models
- Docker containerisation
- Model comparison and selection

## Dataset

Dataset: `data/proj_machine_readings.csv`

The dataset contains:
- 100,000 records
- 50 machines
- 13 columns
- Target variable: `maintenance_required`

Sensor features include:
- temperature
- vibration
- humidity
- pressure
- energy_consumption

The original dataset is preserved without modification.

## Members

- Ethan Kwan Hong Wei 
- Loo Yan Yi 
- Seah Wei Jie 

## Project Structure

```text
EGT309-JaydenInParis/
├── data/
│   └── proj_machine_readings.csv
├── eda.ipynb
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── run.sh
└── README.md
```

## Requirements

- Docker Desktop (or Docker Engine)
- Git
- Web browser

Python dependencies are installed automatically inside Docker.

## Running the Project

### 1. Clone the repository

```bash
git clone <repository-url>
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

EDA findings will guide preprocessing and model selection.

## Machine Learning

Planned models:
1. Logistic Regression
2. Random Forest
3. Histogram-based Gradient Boosting

Models will be evaluated using appropriate classification metrics, including precision, recall, F1-score and PR-AUC.

## Project Status

- [x] Initial EDA
- [x] Docker environment setup
- [x] GitHub repository setup
- [ ] Data preprocessing
- [ ] Model training and evaluation
- [ ] Final model selection
- [ ] Final documentation and submission
```
