# House Price Prediction Using Machine Learning

## Overview

This project develops a machine learning model to predict house prices based on various property characteristics such as living area, number of bedrooms, bathrooms, condition, renovation status, and location zone.

The project includes data preprocessing, exploratory data analysis, model comparison, performance evaluation, and a Streamlit-based web application for real-time predictions.

---

## Objectives

* Analyze housing data and identify important features affecting price.
* Build and evaluate multiple regression models.
* Compare model performance using the R² metric.
* Deploy the best-performing model through an interactive web application.

---

## Dataset Features

The following features were used for prediction:

| Feature           | Description                        |
| ----------------- | ---------------------------------- |
| bedrooms          | Number of bedrooms                 |
| grade             | Overall quality grade of the house |
| has_basement      | Basement availability              |
| living_in_m2      | Living area in square meters       |
| renovated         | Renovation status                  |
| nice_view         | Availability of a good view        |
| perfect_condition | House condition                    |
| real_bathrooms    | Number of bathrooms                |
| has_lavatory      | Lavatory availability              |
| single_floor      | Single-floor property indicator    |
| month             | Month of sale                      |
| quartile_zone     | Location zone category             |

### Target Variable

* price

---

## Exploratory Data Analysis

The dataset was analyzed to understand feature distributions and relationships with house prices.

### Correlation Heatmap

[Insert Correlation Heatmap Screenshot Here]

### Feature Relationship Analysis

[Insert Scatter Plot Screenshot Here]

### Feature Importance Analysis

[Insert Feature Importance Screenshot Here]

---

## Machine Learning Models Evaluated

1. Linear Regression
2. Random Forest Regressor
3. Extra Trees Regressor
4. Gradient Boosting Regressor
5. XGBoost Regressor

---

## Model Performance

| Model             | R² Score |
| ----------------- | -------- |
| Linear Regression | 0.7532   |
| Random Forest     | 0.7463   |
| Extra Trees       |0.7205    |
| Gradient Boosting | 0.7746   |
| XGBoost           |0.7539    |

### Best Performing Model

**Gradient Boosting Regressor**

R² Score: **0.7746**

The model explains approximately 77% of the variance in housing prices.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Matplotlib
* Seaborn
* Streamlit
* Joblib

---

## Project Structure

```text
house_price_prediction/
│
├── app.py
├── house_price.py
├── house.csv
├── house_price_model.pkl
├── requirements.txt
├── README.md

---

## Application Interface

### Prediction Dashboard

[Insert Streamlit Application Screenshot Here]

### Sample Prediction Output

[Insert Prediction Result Screenshot Here]

---

## Installation

Clone the repository:

```bash
git clone https://github.com/krisheka-crypto/house_price_prediction.git
cd house_price_prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---
## screenshots
<img width="1600" height="849" alt="image" src="https://github.com/user-attachments/assets/5757efb5-b04a-4d05-a011-ba839f8909f7" />

<img width="1600" height="850" alt="image" src="https://github.com/user-attachments/assets/bf0ef803-dc19-4de1-bc69-8b7a8fed3bd7" />

<img width="1600" height="850" alt="image" src="https://github.com/user-attachments/assets/e95fbd07-31cb-43ca-99ee-31e1a14fcb3e" />

<img width="1600" height="854" alt="image" src="https://github.com/user-attachments/assets/cedc3777-7922-4d83-87be-5eb3707d2883" />

<img width="1600" height="851" alt="image" src="https://github.com/user-attachments/assets/c400bbde-18f6-48ef-abea-88e9e939aef8" />



## Results

* Successfully developed a house price prediction system.
* Compared multiple regression algorithms.
* Achieved an R² score of 0.7746 using Gradient Boosting Regressor.
* Built an interactive Streamlit application for real-time predictions.

---

## Future Enhancements

* Hyperparameter tuning for improved accuracy.
* Deployment on Streamlit Cloud.
* Advanced feature engineering.
* Interactive data visualization dashboard.

---

## Author

Krisheka

Bachelor of Engineering (Electronics and Communication Engineering)

Machine Learning and Artificial Intelligence Enthusiast
