# Bicycle rental demand forecast

This repository contains an exploratory project focused on predicting hourly bike sharing demand using weather and time-related features. The project showcases data processing, feature engineering, and model development using interpretable and scalable machine learning approaches.

## Project Overview

The goal is to build predictive models that capture the relationships between environmental factors and bike rental demand, with a focus on balancing model interpretability and predictive performance.

## Directory Structure

```
notebooks/
├── eda_cleaning.ipynb # Jupyter notebook for exploratory data analysis and cleaning

src/
├── data/ # Data handling utilities
│ ├── _load_data.py # Data loading functionality
│ └── _sample_split.py # Data splitting functionality
│
├── evaluation/ # Model evaluation components
│ ├── _cv_score.py # Cross-validation scoring
│ └── _evaluate_prediction.py # Prediction evaluation
│
├── feature_engineering/ # Feature transformation tools
│ ├── _discretizer.py # Discretization utility
│ └── _log_transformer.py # Log transformation utility
│
├── plotting/ # Visualization utilities
│ ├── _exploratory_plotting.py # EDA visualizations
│ ├── _interpret_plotting.py # Model interpretation plots
│ └── _model_plotting.py # Model performance plots
│
└── model_training.py # Main model training script

tests/
├── test_discretizer.py # Discretizer tests
└── test_log_transformer.py # Log transformer tests
```
## Installation

1. Create and activate a virtual environment:

   ```
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install the project locally:
   ```
   pip install -e .
   ```
## Usage
Run the main training pipeline with:
   ```
   python src/model_training.py
   ```
This will execute the full workflow including data loading, feature engineering, model training, and evaluation.

