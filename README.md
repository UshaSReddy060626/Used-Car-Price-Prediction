# Used Car Price Prediction (India)
### Machine Learning-Based Used Car Valuation Using LightGBM and Streamlit

## 1. Project Overview

The **Used Car Price Prediction** project is a machine learning-based application designed to estimate the resale value of pre-owned vehicles in the Indian automobile market.

The system analyzes multiple vehicle attributes, including manufacturer, model, manufacturing year, vehicle age, distance driven, engine specifications, fuel type, transmission, ownership history, and geographical location to generate estimated resale prices.

The project implements an end-to-end machine learning workflow, covering data preprocessing, feature engineering, model development, comparative evaluation, and hyperparameter optimization.

Six machine learning models are evaluated using cross-validation and an independent hold-out test dataset. The optimized **LightGBM model** is selected as the final prediction model based on its evaluation performance.

The trained model is integrated into an interactive **Streamlit web application**, allowing users to estimate individual vehicle prices, upload datasets for batch predictions, explore market trends, and inspect model performance.

---

## 2. Problem Statement

Determining the resale value of a used car can be challenging because vehicle prices depend on several factors, including brand reputation, vehicle age, mileage, engine specifications, and market conditions.

Traditional valuation methods may rely on manual estimates or generalized pricing information, which may not accurately reflect the characteristics of an individual vehicle.

This project addresses the problem by developing a machine learning-based prediction system that:

- Analyzes multiple vehicle-related attributes.
- Identifies relationships between vehicle characteristics and resale prices.
- Compares different regression algorithms to identify a suitable prediction model.
- Optimizes model performance through hyperparameter tuning.
- Provides automated price estimates through an interactive web application.

**Proposed Solution:** An end-to-end machine learning application that predicts used car resale prices using vehicle specifications and historical pricing data.

---

## 3. Project Objectives

The primary objectives of this project are:

- Develop a regression-based machine learning model for used car price prediction.
- Perform data cleaning and preprocessing to handle inconsistent and missing values.
- Apply feature engineering to extract meaningful information from vehicle attributes.
- Compare multiple machine learning algorithms.
- Perform hyperparameter tuning to improve prediction performance.
- Evaluate models using R², MAE, RMSE, and percentage-based error metrics.
- Deploy the trained model through a Streamlit dashboard.
- Support individual and batch vehicle price predictions.
- Provide visualizations and model insights to improve prediction interpretability.

---

## 4. Key Features

| Feature | Description |
|---|---|
| Vehicle Price Prediction | Estimates the resale price of pre-owned vehicles. |
| Data Preprocessing | Handles missing values, invalid records, and inconsistent data. |
| Feature Engineering | Creates additional predictive features from existing vehicle attributes. |
| Multiple Model Comparison | Evaluates six regression models. |
| Hyperparameter Tuning | Optimizes XGBoost and LightGBM models. |
| Individual Price Estimation | Generates predictions for individual vehicle specifications. |
| Batch Predictions | Supports CSV uploads for multiple vehicle predictions. |
| Market Trend Analysis | Displays price trends based on vehicle age and distance driven. |
| Model Evaluation | Presents performance metrics for different algorithms. |
| Feature Importance | Identifies attributes contributing to model predictions. |
| Price Range Estimation | Provides estimated fair-price and empirical prediction ranges. |
| CSV Export | Allows users to download batch prediction results. |

---

## 5. Dataset Description

The project uses a structured used-car dataset provided as part of a course capstone.

The dataset contains information about vehicle specifications and corresponding resale prices in the Indian automobile market.

### Dataset Files

| File | Description |
|---|---|
| `Cap_Training_Data_2025.csv` | Training dataset used for model development. |
| `Cap_Test_Data_2025.csv` | Test dataset used to generate final predictions. |
| `Cap_Sample_Submission_2025.csv` | Reference format for prediction submission. |

### Important Dataset Attributes

The dataset includes the following vehicle-related features:

- **Maker:** Manufacturer or brand of the vehicle.
- **Model:** Specific vehicle model.
- **Location:** Geographical location associated with the vehicle.
- **Distance:** Total distance driven in kilometres.
- **Owner Type:** Ownership category of the vehicle.
- **Manufacture Year:** Year in which the vehicle was manufactured.
- **Age of Car:** Current age of the vehicle.
- **Engine Displacement:** Engine capacity measured in cubic centimetres.
- **Engine Power:** Engine power output.
- **Body Type:** Vehicle body classification.
- **Vroom Audit Rating:** Vehicle audit rating.
- **Transmission:** Manual or automatic transmission.
- **Door Count:** Number of doors.
- **Seat Count:** Number of seats.
- **Fuel Type:** Petrol, diesel, or other available fuel categories.

The target variable is the vehicle's resale price.

---

## 6. Data Preprocessing and Feature Engineering

Data preprocessing is an important stage of the project because raw vehicle datasets may contain missing values, inconsistent entries, and unrealistic measurements.

The project implements a structured preprocessing pipeline to improve data quality and prepare the features for machine learning.

### 6.1 Data Cleaning

The system identifies unrealistic vehicle specifications and masks invalid values rather than immediately deleting entire records.

The implemented validation rules include:

- Engine displacement below 600 cc or above 6,500 cc.
- Engine power below 20 or above 500.
- Distance driven exceeding 500,000 km.
- Zero recorded distance for vehicles aged two years or older.

These checks help reduce the influence of unrealistic input values on model training.

### 6.2 Missing Value Imputation

Missing numerical and categorical values are handled using statistics learned exclusively from the training dataset.

The imputation strategy includes:

- Distance imputation based on vehicle age.
- Engine displacement imputation based on model and fuel type.
- Engine power imputation based on model and engine displacement.
- Door and seat count imputation based on vehicle model.

Additional indicator columns are created to identify values that were originally missing.

This allows the model to retain potentially useful information about missingness.

### 6.3 Feature Engineering

Feature engineering transforms existing vehicle attributes into additional variables that may improve prediction accuracy.

The project introduces the following derived features:

| Engineered Feature | Purpose |
|---|---|
| Kilometres per Year | Measures average annual vehicle usage. |
| Log Distance | Reduces the effect of highly skewed mileage values. |
| Power per Litre | Represents engine power relative to engine displacement. |
| Maker-Model Combination | Creates a combined manufacturer and model identifier. |
| Ordinal Owner Type | Converts ownership categories into numerical representations. |
| Binary Transmission | Encodes transmission categories. |
| Binary Fuel | Encodes fuel-related categories. |
| One-Hot Encoded Features | Represents manufacturer, model, and location categories. |

The combined manufacturer-model feature helps distinguish vehicles that share similar model names across different manufacturers.

### 6.4 Data Leakage Prevention

The preprocessing operations are integrated into a Scikit-learn pipeline.

This ensures that imputation statistics and feature transformations are learned from the training portion of each cross-validation fold rather than from the entire dataset.

The independent hold-out test dataset is not used for training or hyperparameter tuning.

This helps provide a more reliable estimate of model generalization performance.

---

## 7. Machine Learning Models

The project evaluates six regression models to identify the most suitable algorithm for predicting used car prices.

The models include:

| Model | Description |
|---|---|
| Ridge Regression | Linear regression with L2 regularization to control model complexity. |
| Random Forest | Ensemble learning algorithm that combines multiple decision trees. |
| XGBoost | Gradient boosting framework designed for efficient predictive modelling. |
| Tuned XGBoost | XGBoost with optimized hyperparameters. |
| LightGBM | Gradient boosting framework using leaf-wise tree growth for efficient learning. |
| Tuned LightGBM | Optimized LightGBM model selected for final deployment. |

### 7.1 Model Training

The models are trained using the cleaned and engineered vehicle attributes.

The project applies five-fold cross-validation to compare model performance across different training and validation splits.

This provides a more comprehensive evaluation than relying on a single training-validation split.

### 7.2 Hyperparameter Tuning

Hyperparameter optimization is performed for XGBoost and LightGBM to improve their predictive performance.

The tuning process identifies suitable model configurations based on validation performance.

The tuned models are subsequently evaluated using the independent hold-out dataset.

### 7.3 Target Transformation

The prediction target is transformed using:

```python
log(1 + Price)
```

This logarithmic transformation helps reduce the impact of highly skewed vehicle prices during model training.

The predicted values are converted back into the original rupee scale before calculating evaluation metrics and displaying results.

---

## 8. Model Evaluation and Results

The final models are evaluated using an independent 20% hold-out test dataset that is not used during model training or hyperparameter tuning.

The tuned LightGBM model is selected as the deployed model.

### 8.1 Evaluation Metrics

The project uses the following regression evaluation metrics:

**R² Score (Coefficient of Determination)**

Measures how effectively the model explains variations in actual vehicle prices. Higher values indicate better predictive performance.

**Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted vehicle prices.

**Root Mean Squared Error (RMSE)**

Measures prediction error while assigning greater penalties to larger errors.

**Median Absolute Percentage Error (MdAPE)**

Measures the median percentage difference between predicted and actual vehicle prices.

### 8.2 Final Model Performance

<text id="0" weight="bold">Deployed Model: Tuned LightGBM</text>

| Evaluation Metric | Result |
|---|---:|
| R² Score | 0.946 |
| Mean Absolute Error | ₹1.10 lakh |
| Root Mean Squared Error | ₹1.96 lakh |
| Median Absolute Percentage Error | 8.3% |
| Predictions within ±10% of actual price | 57.7% |
| Predictions within ±20% of actual price | 84.9% |

**Performance Interpretation:**

- The R² score of 0.946 indicates that the model explains approximately 94.6% of the variance in the hold-out target values.
- The average absolute prediction error is approximately ₹1.10 lakh.
- The median percentage prediction error is 8.3%.
- Approximately 84.9% of predictions fall within 20% of their actual prices.

These results indicate strong predictive performance on the available hold-out dataset. They should not be interpreted as a guarantee of equivalent accuracy across the entire used-car market.

---

## 9. Streamlit Web Application

The trained LightGBM model is integrated into an interactive web application developed using Streamlit.

The dashboard provides multiple functionalities for estimating vehicle prices, analysing market trends, and inspecting model performance.

### 9.1 Estimate

Allows users to enter individual vehicle specifications through the sidebar and obtain an estimated resale price.

The section provides:

- Predicted vehicle price.
- Estimated fair-price range.
- Empirical 80% prediction range.
- Visualizations showing how vehicle age and distance driven influence estimated prices.

The empirical range is calculated using the 10th and 90th percentiles of actual-to-predicted price ratios from the hold-out dataset.

The calibration factors are:

```text
Lower Bound = 0.85
Upper Bound = 1.20
```

These are empirical calibration factors, not guaranteed confidence intervals.

### 9.2 Batch Predictions

Allows users to upload a CSV file containing multiple vehicle records.

The application:

- Accepts structured vehicle information.
- Applies the same preprocessing pipeline used during training.
- Generates predictions for multiple vehicles.
- Displays prediction summary statistics.
- Provides downloadable prediction results.

An optional `ID` column is preserved in the generated output.

### 9.3 Market Trends

Provides visual insights into used-car pricing patterns.

Users can explore:

- Median resale prices across different vehicle ages.
- Price variations based on distance driven.
- Market trends for selected manufacturers.

These visualizations help users understand how vehicle depreciation and usage relate to resale prices in the available dataset.

### 9.4 Model Details

Provides information about the machine learning models and their evaluation.

The section includes:

- Hold-out performance comparison of all six models.
- Feature importance visualizations.
- Model evaluation metrics.
- Information about model limitations.

This section improves transparency by allowing users to inspect the performance and characteristics of the deployed model.

---

## 10. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computations |
| Scikit-learn | Regression models, preprocessing pipelines, cross-validation, and evaluation |
| LightGBM | Final optimized prediction model |
| XGBoost | Gradient boosting model comparison |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Joblib | Saving and loading the trained model |
| Streamlit | Interactive web application |
| Jupyter Notebook | Exploratory analysis and model development |
| Pytest | Automated project testing |
| CSV | Dataset storage and prediction exports |

---

## 11. Project Structure

```text
car-price-prediction/
│
├── app.py
│   └── Streamlit dashboard for price estimation,
│       batch predictions, market trends, and model details
│
├── car_utils.py
│   └── Shared data cleaning and feature engineering functions
│
├── notebooks/
│   └── Car_Price_Prediction.ipynb
│       └── Exploratory data analysis, preprocessing,
│           model training, evaluation, and model export
│
├── data/
│   ├── Cap_Training_Data_2025.csv
│   ├── Cap_Test_Data_2025.csv
│   └── Cap_Sample_Submission_2025.csv
│
├── models/
│   └── car_price_model.joblib
│       └── Trained LightGBM model and dashboard metadata
│
├── outputs/
│   └── submission.csv
│       └── Generated test-dataset predictions
│
├── tests/
│   └── test_project.py
│       └── Automated project tests
│
├── .streamlit/
│   └── config.toml
│       └── Dashboard theme configuration
│
├── requirements.txt
│   └── Runtime dependencies
│
├── requirements-dev.txt
│   └── Development and testing dependencies
│
├── LICENSE
│
└── README.md
```

---

## 12. Installation and Execution

### Prerequisites

- Python 3.11 or newer.
- pip package manager.
- Git.
- Required project datasets and trained model files.

### Step 1: Clone the Repository

```bash
git clone <repository-url>
```

### Step 2: Navigate to the Project Directory

```bash
cd car-price-prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Launch the Streamlit Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

The dashboard can now be accessed through a web browser.

### 12.1 Running Tests

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Execute the automated tests:

```bash
pytest
```

### 12.2 Running the Notebook

Open the following notebook using Jupyter Notebook or Visual Studio Code:

```text
notebooks/Car_Price_Prediction.ipynb
```

Select the `.venv` environment as the notebook kernel.

Running all notebook cells retrains the evaluated models and regenerates:

- `models/car_price_model.joblib`
- `outputs/submission.csv`

The complete training and evaluation process takes approximately 20 minutes on a single CPU core.

**Important:** The saved model was created using Scikit-learn 1.8.0 and Pandas 3.0. The project pins these versions in `requirements.txt` to maintain compatibility. If different versions are used, retrain the model using the notebook.

---

## 13. Batch Prediction Input Format

The batch prediction functionality accepts CSV files containing the required vehicle attributes.

The expected columns are:

```text
Maker
model
Location
Distance
Owner Type
manufacture_year
Age of car
engine_displacement
engine_power
body_type
Vroom Audit Rating
transmission
door_count
seat_count
fuel_type
```

An optional `ID` column can also be included and is preserved in the prediction output.

The application automatically handles trailing spaces in column names, including the `Distance ` field present in the original dataset.

A sample input CSV is provided through the dashboard for reference.

---

## 14. Limitations

Despite achieving strong evaluation results, the project has certain limitations:

1. **Dataset Characteristics:** The available dataset appears to contain some synthetic or unrealistic patterns, such as vehicles manufactured before 1980 and identical prices across certain ownership categories.
2. **Market Generalization:** Performance on the available dataset may differ from actual used-car market conditions.
3. **Feature Contribution:** Location, ownership type, and audit rating provide relatively limited predictive information in the current dataset.
4. **Rare Vehicle Models:** Predictions for uncommon manufacturers and models, such as Maserati, may have higher errors because of limited representation.
5. **Extreme Vehicle Attributes:** Very old vehicles and vehicles with unusually high mileage may produce less reliable estimates.
6. **Price Estimation:** The generated prices are indicative predictions and should not be considered official vehicle valuations.

---

## 15. Future Enhancements

The project can be extended through the following improvements:

- **Larger Real-World Datasets:** Integrate more recent and diverse used-car listings to improve generalization.
- **Advanced Feature Engineering:** Incorporate depreciation patterns, brand-specific resale trends, and additional vehicle attributes.
- **Model Optimization:** Explore additional boosting algorithms and ensemble techniques.
- **Market Data Integration:** Incorporate real-time vehicle listing information to reflect changing market conditions.
- **Improved Price Calibration:** Develop more robust prediction intervals across different vehicle categories.
- **Cloud Deployment:** Deploy the application on a scalable cloud platform.
- **Database Integration:** Store vehicle information and prediction history in a structured database.
- **Enhanced Explainability:** Introduce SHAP-based explanations to understand individual vehicle price predictions.

---

## 16. Conclusion

The **Used Car Price Prediction (India)** project demonstrates the application of machine learning techniques to estimate the resale value of pre-owned vehicles using multiple vehicle-related attributes.

The project covers the complete machine learning lifecycle, including data preprocessing, feature engineering, model comparison, hyperparameter optimization, independent model evaluation, and application deployment.

By comparing six regression models and selecting an optimized LightGBM model, the project achieves an R² score of 0.946 and a median absolute percentage error of 8.3% on the hold-out dataset.

The Streamlit dashboard makes the trained model accessible through individual price estimation, batch predictions, market trend visualizations, and model performance inspection.

**Core Workflow:**

```text
       RAW VEHICLE DATA
               |
               v
        DATA CLEANING
               |
               v
      FEATURE ENGINEERING
               |
               v
       MODEL DEVELOPMENT
               |
               v
    CROSS-VALIDATION AND TUNING
               |
               v
      MODEL COMPARISON
               |
               v
       OPTIMIZED LIGHTGBM
               |
               v
       MODEL EVALUATION
               |
               v
      STREAMLIT DEPLOYMENT
               |
               v
       USED CAR PRICE
          PREDICTION
```

**Project Focus:** Developing a reliable, data-driven vehicle valuation system through machine learning, predictive modelling, and interactive application development.