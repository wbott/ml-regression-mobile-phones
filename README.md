
# 📱 Mobile Phones: What Are You Paying For?

![iPhone](pexels-photo-1092644-s.jpg)

This project investigates the relationship between smartphone features and their prices using various regression techniques. It aims to answer the question: _Which features influence mobile phone pricing the most, and how accurately can we predict price using those features?_

---

## 🔧 Setup

Install the required packages:

```bash
pip install kaggle kagglehub ipkernel
pip install ipywidgets --upgrade
pip install jupyterlab_widgets
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
```

---

## 🎯 Objective

Explore and compare multiple regression models to determine the most effective one for predicting mobile phone prices.

---

## 📊 Workflow

### 1. 📁 Dataset Overview
- Load and describe the dataset
- Understand structure, datatypes, and missing values

### 2. 📈 Exploratory Data Analysis (EDA)
- Visualize the distribution of the target variable (Price)
- Analyze individual features
- Explore feature-target relationships
- Check for multicollinearity using VIF and correlation heatmaps

### 3. 🧹 Data Preprocessing
- Handle missing or inconsistent values
- Normalize / transform variables as needed
- Split into training and test sets

### 4. 🤖 Model Training & Evaluation

**Models Explored:**
- 🔹 Simple Linear Regression
- 🔹 Polynomial Regression
- 🔹 Random Forest Regressor
- 🔹 Regularized Regression (Ridge/Lasso)

**Evaluation Metrics:**
- 📉 Mean Absolute Error (MAE)
- 📉 Mean Squared Error (MSE)
- 📈 R-squared (R²)
- 📉 Residual plots and Actual vs Predicted comparisons

---

## 🧠 Key Insights

- Multicollinearity was present in features such as `Ratings` and `Battery_Power`
- Random Forest provided the best predictive performance with R² > 0.95
- Visualizations revealed a strong fit with minimal residual error

---

## 📌 Future Work

- Feature selection and dimensionality reduction (PCA)
- Deploy model via a Flask app or Streamlit dashboard
- Integration with real-time pricing APIs for prediction

---

## 🙌 Acknowledgments

- Dataset sourced via Kaggle
- Visualization powered by Matplotlib and Seaborn
- Models built with Scikit-learn and Statsmodels

---

## 📂 License

This project is open-source under the [MIT License](LICENSE).
