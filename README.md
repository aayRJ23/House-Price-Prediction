# Housing Price Prediction

This project aims to predict housing prices in Boston using machine learning techniques, particularly XGBoost Regression and Linear Regression. The dataset used for this project is the Boston Housing dataset, which contains various features related to housing in different suburbs of Boston. The goal is to build a model that can accurately predict the median value of owner-occupied homes.

## Dataset

The dataset used in this project is loaded from the `boston.csv` file. It contains the following columns:

- CRIM: Per capita crime rate by town
- ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS: Proportion of non-retail business acres per town
- CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- NOX: Nitric oxides concentration (parts per 10 million)
- RM: Average number of rooms per dwelling
- AGE: Proportion of owner-occupied units built prior to 1940
- DIS: Weighted distances to five Boston employment centers
- RAD: Index of accessibility to radial highways
- TAX: Full-value property tax rate per $10,000
- PTRATIO: Pupil-teacher ratio by town
- B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT: Percentage of lower status of the population
- MEDV: Median value of owner-occupied homes in $1000s (Target Variable)

## Analysis

1. **Data Exploration and Preprocessing:**
   - The dataset is explored using various methods such as `head()`, `tail()`, `shape`, `isnull().sum()`, and `describe()` to understand its structure, check for missing values, and obtain summary statistics.
   - A heatmap is plotted to visualize the correlation between different features and the target variable (`MEDV`).

2. **Data Splitting:**
   - The dataset is split into training and testing sets using the `train_test_split` function from `sklearn.model_selection`.

3. **Model Building:**
   - Two models are implemented for prediction: XGBoost Regression and Linear Regression.
   - XGBoost Regression is implemented using the `XGBRegressor` class from the `xgboost` library. The model is trained on the training data.
   - Predictions are made on both the training and test sets.

4. **Model Evaluation:**
   - Evaluation metrics such as R-squared error and Mean Absolute Error (MAE) are calculated to assess the performance of the models.
   - Actual vs. Predicted price plots are generated for both training and test sets to visualize the model performance.

## Files Included
- `boston.csv`: Dataset containing housing-related features and median house prices.
- `README.md`: This file provides an overview of the project, its objectives, dataset, analysis, and model implementation.
- `HousePricePrediction.ipynb`: Jupyter Notebook containing the Python code for data loading, preprocessing, model building, and evaluation.

## Instructions to Run
1. Clone the repository to your local machine.
2. Ensure that Python and necessary libraries (NumPy, pandas, matplotlib, seaborn, scikit-learn, xgboost) are installed.
3. Run the `HousePricePrediction.ipynb` notebook in a Jupyter environment or any compatible IDE.

## Requirements
- Python 3.x
- NumPy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost

## Conclusion
This project demonstrates the implementation of machine learning algorithms for predicting housing prices based on various features. The models can be further optimized and tuned to improve prediction accuracy. Additionally, feature engineering and selection techniques can be explored to enhance model performance further.
