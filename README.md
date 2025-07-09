
# 🚗 Car Price Prediction with Machine Learning

## Project Overview

This project builds a machine learning model to predict the **selling price of used cars** based on key factors like current price, age, kilometers driven, fuel type, transmission, and more. It follows a full end-to-end ML pipeline using Python and culminates in a working **Streamlit web app**.

---

##  Step 1: Objective

* Predict resale value of cars using historical data.
* Understand feature impact on price.
* Build a deployable web app for real-time prediction.

---

##  Step 2: Data Preprocessing

**Dataset**: `car data.xlsx`

Tasks performed:

* Loaded data using `pandas`
* Renamed `Driven_kms` → `Driven_Kms`
* Converted `Year` to `Car_Age` using `2025 - Year`
* Dropped `Car_Name` and original `Year`
* Label encoded categorical columns:

  * `Fuel_Type`: Petrol, Diesel, CNG → 2, 0, 1
  * `Transmission`: Manual → 1, Automatic → 0
  * `Selling_type`: Dealer, Individual, Trustmark Dealer → 0, 1, 2
* Removed outliers from `Driven_Kms` and `Owner`

Result: Cleaned dataset ready for visualization and modeling

---

##  Step 3: Exploratory Data Analysis (EDA)

Performed using `matplotlib` and `seaborn`:

* Histogram: `Selling_Price` distribution
* Countplots: `Fuel_Type`, `Transmission`, `Seller_Type`
* Scatter: `Present_Price` vs `Selling_Price`
* Heatmap: Feature correlation
* Boxplots: `Selling_Price` vs `Fuel_Type` and `Owner`

Insights:

* Higher `Present_Price` → higher `Selling_Price`
* More owners → lower price
* Petrol cars dominate the dataset

---

##  Step 4: Feature Selection

Used `ExtraTreesRegressor` to determine feature importance.
Top important features:

1. `Present_Price`
2. `Car_Age`
3. `Fuel_Type`
4. `Kms_Driven`

Final selected `X` and `y` for training:

* `X_final` = all encoded features
* `y_final` = `Selling_Price`

---

##  Step 5: Model Building

Used `train_test_split()` to create training and test data.

Trained models:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Decision Tree Regressor
* Random Forest Regressor
* (Optional) XGBoost Regressor

---

##  Step 6: Model Evaluation

Metrics used:

* R² Score
* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)

Each model was evaluated and compared.
**Random Forest** gave the best performance.

Also plotted:

* Predicted vs Actual `Selling_Price`

---

##  Step 7: Hyperparameter Tuning

Tuned `RandomForestRegressor` using `GridSearchCV`
Grid parameters:

* `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`

Best parameters were selected and re-evaluated on test data.

---

##  Step 8: Save the Model

Used `joblib` to save the best trained model:

```python
joblib.dump(best_rf, 'car_price_model.pkl')
```

Model can be reused later for inference:

```python
model = joblib.load('car_price_model.pkl')
```

---

##  Step 9: Web App with Streamlit

Built a web app using `Streamlit` with the following features:

* UI to input car details (price, age, fuel, etc.)
* Model loaded using `joblib`
* Prediction displayed live in browser

###  Launch Command:

```bash
streamlit run app.py
```

URL: `http://localhost:8501`

---

##  Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn, Joblib
* Matplotlib, Seaborn
* Streamlit

---

##  Conclusion

This project demonstrates:

* End-to-end machine learning pipeline
* Data cleaning, visualization, modeling
* Hyperparameter tuning and deployment
* Practical use of `Streamlit` to share your ML model as a real-time web application

---


