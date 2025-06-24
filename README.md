
# 📱 Mobile Phones: What Are You Paying For?

![iPhone](pexels-photo-1092644-s.jpg)

This project investigates the relationship between smartphone features and their prices using various regression techniques. It aims to answer the question: _Which features influence mobile phone pricing the most, and how accurately can we predict price using those features?_

---

## 🔧 Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels kaggle jupyter ipykernel
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
- 🔹 Ordinary Least Squares (OLS) Regression
- 🔹 Random Forest Regressor
- 🔹 Ridge Regression (for multicollinearity)

**Evaluation Metrics:**
- 📉 Mean Absolute Error (MAE)
- 📉 Mean Squared Error (MSE)
- 📈 R-squared (R²)
- 📊 Variance Inflation Factor (VIF) for multicollinearity detection
- 📉 Actual vs Predicted scatter plots

---

## 🧠 Key Insights

- **Dataset**: 807 mobile phones with 8 features (Ratings, RAM, ROM, Mobile_Size, Primary_Cam, Selfi_Cam, Battery_Power, Price)
- **Data Preprocessing**: Applied log transformation to stabilize price variance and handle right-skewed distribution
- **Multicollinearity**: High VIF values detected in Ratings (40.5), Battery_Power (18.4), and Primary_Cam (18.0)
- **Model Performance**: 
  - OLS Regression: R² = 0.67
  - Random Forest: R² = 0.95 (best performance)
- **Key Predictors**: Ratings (0.69), ROM (0.61), and Battery_Power (0.55) show strongest correlation with price
- **Surprising Findings**: Primary_Cam shows negative correlation (-0.27) with price

---

## 📌 Future Work

- Feature selection and dimensionality reduction (PCA)
- Deploy model via a Flask app or Streamlit dashboard
- Integration with real-time pricing APIs for prediction

---

## 🛠️ Technologies & Tools

### Core Libraries
- **NumPy** - Numerical computations and array operations
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Data visualization and plotting
- **Seaborn** - Statistical data visualization
- **Scikit-learn** - Machine learning algorithms and metrics
- **Statsmodels** - Statistical modeling and analysis

### Development Environment
- **Jupyter Notebook** - Interactive development and analysis
- **Python 3.x** - Programming language
- **Kaggle API** - Dataset download and management

### Statistical Analysis
- **Variance Inflation Factor (VIF)** - Multicollinearity detection
- **Correlation Analysis** - Feature relationship analysis
- **Log Transformation** - Data normalization
- **Train-Test Split** - Model validation

### Machine Learning Models
- **Ordinary Least Squares (OLS)** - Linear regression with statistical inference
- **Random Forest Regressor** - Ensemble learning method
- **Ridge Regression** - Regularized linear regression
- **StandardScaler** - Feature scaling and normalization

---

## 🙌 Acknowledgments

- Dataset: [Mobile Phone Price Prediction](https://www.kaggle.com/datasets/ganjerlawrence/mobile-phone-price-prediction-cleaned-dataset) by Ganjerlawrence on Kaggle
- Visualization powered by Matplotlib and Seaborn
- Models built with Scikit-learn and Statsmodels

---

## 📂 License

This project is open-source under the [MIT License](LICENSE).
