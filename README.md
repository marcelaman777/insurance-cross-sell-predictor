# Health Insurance Cross-Selling Prediction - EDA & CatBoost Modeling

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-brightgreen)
![Model](https://img.shields.io/badge/Model-CatBoost-red)
![Analysis](https://img.shields.io/badge/Analysis-EDA-yellowgreen)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Project Overview

This project performs comprehensive Exploratory Data Analysis (EDA) and builds a CatBoost classification model for an insurance company's cross-selling prediction challenge. The goal is to predict whether existing health insurance customers would be interested in purchasing vehicle insurance based on their demographic and historical data.

**Business Context**: An insurance company that has provided health insurance to its customers needs to build a model to predict whether last year's policyholders would also be interested in purchasing vehicle insurance.

## 🎯 Objective

Predict which customers respond positively to an automobile insurance offer based on 10 customer features.

## 📊 Dataset Features

### Categorical Features
- **Gender**: Gender of the customer
- **Driving_License**: 0 - No DL, 1 - Has DL
- **Region_Code**: Unique code for the region
- **Previously_Insured**: 1 - Has Vehicle Insurance, 0 - Doesn't have
- **Vehicle_Age**: Age of the Vehicle (Ordinal: < 1 Year, 1-2 Year, > 2 Years)
- **Vehicle_Damage**: 1 - Vehicle damaged in past, 0 - No damage
- **Policy_Sales_Channel**: Anonymized code for outreach channel

### Numerical Features
- **Age**: Age of the customer (Discrete)
- **Annual_Premium**: Amount customer pays as premium (Continuous)
- **Vintage**: Number of days associated with company (Discrete)

### Target Variable
- **Response**: 1 - Customer is interested, 0 - Customer is not interested

## 🏗️ Project Structure

### Phase 1: Exploratory Data Analysis (EDA)
**File**: `classification-of-insurance-cross-selling-eda.ipynb`

### Phase 2: CatBoost Modeling with Weight of Evidence
**File**: `insurance-selling-weight-of-evidence-catboost.ipynb`

## 🔍 EDA Methodology

This notebook implements a comprehensive EDA approach:

### 1. Univariate Analysis
- Statistical summaries for categorical and numerical features
- Distribution comparisons between train and test sets
- Hypothesis testing (Anderson-Darling, Mann-Whitney U, Proportion tests)
- Box plots and target distribution analysis

### 2. Bivariate & Multivariate Analysis
- Violin plots by response and feature combinations
- Point plots for multi-variable relationships
- Interactive visualizations across multiple dimensions

### 3. Association Measures
- **Spearman Correlation**: For monotonic relationships
- **Cramér's V**: For categorical-categorical associations
- **Mutual Information**: For general dependency between variables

### 4. Advanced Techniques
- **Hierarchical Clustering**: Feature grouping using Ward's method
- **Weight of Evidence (WoE) Transformation**: For categorical feature encoding
- **Factor Analysis for Mixed Data (FAMD)**: Dimensionality reduction for mixed data types

### 📈 Key Insights from EDA

**Strong Correlations Identified (Spearman's method):**
- **Vehicle_Damage ↔ Previously_Insured** (-0.84)
- **Age ↔ Vehicle_Age** (0.83)
- **Age ↔ Policy_Sales_Channel** (-0.66)

**Top Predictive Features:**
- **Previously_Insured** (Strong negative correlation with response)
- **Vehicle_Damage** (Strong positive correlation with response)
- **Vehicle_Age and Age** (Moderate correlations)

**Feature Clusters:**
- **Cluster 1**: Annual_Premium + Region_Code
- **Cluster 2**: Vehicle_Age + Policy_Sales_Channel + Age
- **Cluster 3**: Vehicle_Damage + Previously_Insured
- **Cluster 4**: Vintage + Gender + Driving_License

## 🤖 CatBoost Modeling Implementation

### Feature Engineering

#### Weight of Evidence (WoE) Transformation
Implemented custom `TargetEncodingTransformer` class to convert categorical features into continuous using WoE:

```python
class TargetEncodingTransformer(TransformerMixin, BaseEstimator):
    '''Transform categorical features into continuous ones using Weight of Evidence'''
    
    def calculate_woe(self, X, y):
        '''Calculate Weight of Evidence (WoE)'''
        eps = 1e-10  # small value to avoid division by zero
        grouped = y.groupby(X).agg(['count', 'sum'])
        grouped['non_event'] = grouped['count'] - grouped['sum'] 
        grouped['woe'] = np.log((grouped['sum'] + eps) / (grouped['non_event'] + eps))
        return dict(grouped['woe'])
```

**Features transformed with WoE**:
- `Region_Code_WoE`
- `Policy_Sales_Channel_WoE`

**Data Preprocessing**:
- Ordinal encoding for categorical variables
- Downcasting of `Annual_Premium` to reduce memory usage
- ColumnTransformer for efficient feature processing

### Model Architecture
**CatBoost Classifier Configuration**:
```python
model_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'learning_rate': 0.075,
    'iterations': 10000,
    'depth': 9,
    'random_strength': 0,
    'l2_leaf_reg': 0.5,
    'max_leaves': 512,
    'fold_permutation_block': 64,
    'task_type': 'GPU',
    'random_seed': 42,
    'early_stopping_rounds': 500
}
```

**Training Strategy**:
- **5-Fold Stratified Cross-Validation**
- **Early Stopping** with 500 rounds patience
- **GPU Acceleration** for faster training
- **Ensemble Prediction** by averaging fold predictions

### Performance Results
| Fold | Validation AUC | Best Iteration |
|------|----------------|----------------|
| 1    | 0.89492        | 2375           |
| 2    | 0.89446        | 2639           |
| 3    | 0.89474        | 2190           |
| 4    | 0.89458        | 3156           |
| 5    | 0.89526        | 2567           |

** 📈 Final Model Performance:**
- **Mean Validation AUC**: 0.89479
- **Training Time**: ~4.5 hours (GPU accelerated)
- **Dataset Size**: 11.5M training samples, 7.7M test samples

## 🛠️ Technical Implementation

**Libraries Used:**
- **pandas, numpy** - Data manipulation
- **matplotlib, seaborn** - Visualization
- **scipy, statsmodels** - Statistical testing
- **scikit-learn** - Feature selection and metrics
- **prince** - Factor analysis for mixed data
- **catboost** - Gradient boosting implementation

**Key Features:**
- Automated hypothesis testing between train/test distributions
- Multi-dimensional point plot visualizations
- WoE transformation for categorical feature engineering
- Stratified cross-validation for robust evaluation
- GPU-accelerated training with CatBoost

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels prince catboost
```

## Usage
**Clone this repository:**
```bash
git clone https://github.com/tuusuario/insurance-cross-selling-eda.git
```

**Navigate to the project directory:**
```bash
cd insurance-cross-selling-eda
```

**Run the notebooks in order:**
```bash
# First: Exploratory Data Analysis
jupyter notebook classification-of-insurance-cross-selling-eda.ipynb

# Second: CatBoost Modeling
jupyter notebook insurance-selling-weight-of-evidence-catboost.ipynb
```

## 📊 Evaluation Metric
This project is part of [Kaggle's Playground Series S4E7 competition](https://www.kaggle.com/competitions/playground-series-s4e7). The model performance is evaluated using:

**Area Under the ROC Curve (AUC-ROC)**
- **Range**: 0 to 1 (higher is better)
- **Interpretation:**
  - 0.5 = Random guessing
  - 0.7-0.8 = Good model
  - 0.8-0.9 = Very good model
  - \>0.9 = Excellent model

## 📊 Competition Details
- **Competition**: [Playground Series Season 4 Episode 7](https://www.kaggle.com/competitions/playground-series-s4e7)
- **Evaluation**: AUC-ROC
- **Submission Format**: CSV with customer IDs and predicted probabilities

## 🔮 Future Work
**Model Improvements**
- Hyperparameter Optimization: Bayesian optimization for CatBoost parameters
- Feature Engineering: Create interaction terms based on EDA insights
- Ensemble Methods: Combine CatBoost with other gradient boosting models
- Class Balancing: Address potential class imbalance if present

**Advanced Techniques**
- SHAP Analysis: For model interpretability and feature importance
- Cross-Validation Strategies: Time-series or group-based CV
- Model Calibration: Ensure predicted probabilities are well-calibrated

## 📚 References
**Competition & Data**
- [Kaggle Competition: Playground Series S4E7](https://www.kaggle.com/competitions/playground-series-s4e7)
- [Original Dataset: Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)

**Technical References**
- [CatBoost Documentation](https://catboost.ai/)
- [Weight of Evidence Explained](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)
- [Prince Documentation - FAMD](https://github.com/MaxHalford/prince)

**🌟 Highlights:** This project demonstrates a complete machine learning pipeline from exploratory data analysis to production-ready modeling, featuring advanced techniques like Weight of Evidence transformation and GPU-accelerated CatBoost training on large-scale insurance data.
