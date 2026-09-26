# ✈️ Flight Price Prediction

A Machine Learning project for predicting flight ticket prices using real-world flight data. The project covers data preprocessing, exploratory analysis, multiple regression models, hyperparameter optimization, model evaluation, feature importance analysis, and an interactive Streamlit application.

---

## 📌 Project Overview

Flight ticket prices depend on several factors such as airline, source and destination cities, departure and arrival times, number of stops, travel class, flight duration, and the number of days before departure.

The objective of this project is to build a Machine Learning regression model capable of estimating flight prices from these features.

The project follows an end-to-end Machine Learning workflow:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Encoding
     ↓
Train / Test Split
     ↓
Multiple Regression Models
     ↓
Model Comparison
     ↓
Hyperparameter Optimization
     ↓
Feature Importance
     ↓
Final Model
     ↓
Streamlit Application
```

---

## 🎯 Objectives

* Analyze flight price data.
* Clean and prepare the dataset.
* Transform categorical variables using One-Hot Encoding.
* Train and compare several Machine Learning regression models.
* Evaluate models using standard regression metrics.
* Optimize the Random Forest model using `RandomizedSearchCV`.
* Identify the most important features influencing flight prices.
* Save the final trained model.
* Build an interactive web application using Streamlit.

---

## 📂 Dataset

The project uses the `Clean_Dataset.csv` dataset.

The dataset contains information about flights, including:

* Airline
* Source City
* Destination City
* Departure Time
* Arrival Time
* Stops
* Class
* Duration
* Days Left
* Price

The columns `Unnamed: 0` and `flight` were removed during preprocessing because they were not used as predictive features.

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

### 1. Remove unnecessary columns

```python
data = data.drop('Unnamed: 0', axis=1)
data = data.drop('flight', axis=1)
```

### 2. Encode flight class

Business class was encoded as `1` and Economy class as `0`.

### 3. Encode number of stops

The `stops` feature was converted into numerical values.

### 4. One-Hot Encoding

Categorical variables were transformed using Pandas `get_dummies()`:

* Airline
* Source City
* Destination City
* Arrival Time
* Departure Time

This produced numerical features suitable for Machine Learning models.

---

## 🤖 Machine Learning Models

Several regression algorithms were trained and evaluated:

* Linear Regression
* Polynomial Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor
* Random Forest with RandomizedSearchCV optimization

The models were evaluated on the same test set to make the comparison consistent.

---

## 📊 Model Evaluation

The following metrics were used:

### R² Score

Measures how well the model explains the variance in the target variable.

Higher values indicate better performance.

### MAE — Mean Absolute Error

Measures the average absolute difference between predicted and actual prices.

Lower values are better.

### MSE — Mean Squared Error

Penalizes larger prediction errors more strongly.

Lower values are better.

### RMSE — Root Mean Squared Error

Provides the prediction error in the same unit as the target variable.

Lower values are better.

---

## 🌲 Random Forest Optimization

The Random Forest model was further optimized using `RandomizedSearchCV`.

The search explored parameters including:

```python
param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'max_features': [1.0, 'sqrt']
}
```

The optimized model achieved an R² score of approximately:

```text
R² ≈ 0.986
```

On the test set, the optimized model achieved approximately:

```text
R²   ≈ 0.986
MAE  ≈ 1,084
RMSE ≈ 2,666
```

These values correspond to the current test split and should be interpreted in the context of this dataset and evaluation setup.

---

## 🔎 Feature Importance

Feature importance was extracted from the optimized Random Forest model to identify which variables contributed most to the predictions.

The project includes a visualization of the top 10 features.

Example:

```python
importances = pd.Series(
    best_regressor.feature_importances_,
    index=x_train.columns
)

importances.sort_values(ascending=False).head(10)
```

This provides an interpretable view of the factors used by the Random Forest model.

---

## 💾 Model

The final optimized model is saved using Joblib:

```python
joblib.dump(
    best_regressor,
    'Models/flight_price_model.pkl'
)
```

The saved model is used by the Streamlit application to generate predictions.

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit application where users can enter flight information and receive an estimated ticket price.

The application uses the same feature encoding approach used during model training before passing the data to the trained model.

### Run the application

```bash
streamlit run app.py
```

The application provides:

* Airline selection
* Source city
* Destination city
* Departure time
* Arrival time
* Number of stops
* Flight class
* Flight duration
* Days before departure
* Predicted flight price

---

## 📁 Project Structure

```text
Flight_Price_Prediction/
│
├── app.py
├── main.ipynb
├── Clean_Dataset.csv
├── README.md
├── requirements.txt
├── .gitignore
│
└── Models/
    └── flight_price_model.pkl
```

---

## 🛠️ Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy
* Matplotlib

### Machine Learning

* Scikit-learn
* XGBoost

### Model Optimization

* RandomizedSearchCV

### Model Deployment

* Joblib
* Streamlit

### Development

* Jupyter Notebook
* Git
* GitHub

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Flight_Price_Prediction.git
cd Flight_Price_Prediction
```

Create a virtual environment:

```bash
conda create -n flight_price python=3.11
conda activate flight_price
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📈 Project Highlights

* End-to-end Machine Learning workflow
* Multiple regression algorithms compared
* Random Forest hyperparameter optimization
* R² score of approximately 0.986 on the test set
* Feature importance analysis
* Interactive Streamlit prediction application
* Reproducible Python-based workflow

---


## 👩‍💻 Author

**Isra Nour El Yakine Brahimi**

AI / Computer Vision Engineer
Master's Degree in Embedded Systems Engineering

LinkedIn: `https://linkedin.com/in/isra-nour-el-yakine-b-713a38208`
