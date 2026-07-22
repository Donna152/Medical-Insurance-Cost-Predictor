# 🏥 Medical Insurance Cost Predictor

A machine learning web app that predicts an individual's annual medical insurance charges based on age, sex, BMI, number of children, smoking status, and region. Built with scikit-learn and deployed as an interactive Streamlit app.

**🔗 Live Demo:** [donna7-laptop-price-predictor.streamlit.app](https://donna7-medical-insurance-predictor.streamlit.app)

---

## 📌 Overview

This project builds a regression pipeline that estimates medical insurance charges from personal and demographic attributes. After exploratory analysis of the classic insurance cost dataset, categorical features are encoded and numerical features scaled, seven regression algorithms are trained and compared, and the pipeline is validated for overfitting and stability via K-Fold cross-validation before being bundled into a deployable object.

## ✨ Features

- 🧍 Enter age, sex, BMI, number of children, smoking status, and region
- 💰 Get an instant estimated annual insurance charge
- 📊 Final model selected after comparing 7 regression algorithms on R² and MAE
- ✅ Validated with train/test overfitting checks and 5-fold cross-validation
- 🔗 End-to-end scikit-learn `Pipeline` (encoding + scaling + model) for consistent predictions

## 🖼️ Screenshots

Screenshots of the app in action are available in the [`Results`](./Results) folder.

## 🧠 How It Works

### 1. Dataset

The project uses **`insurance.csv`** (in the [`dataset`](./dataset) folder) — a complete, no-missing-values collection of **1,338 individual records**:

| Column | Description |
|---|---|
| `age` | Age of the primary beneficiary |
| `sex` | Insurance contractor gender (female, male) |
| `bmi` | Body Mass Index (kg/m²) |
| `children` | Number of dependents covered by the insurance plan |
| `smoker` | Smoking status (yes/no) |
| `region` | Beneficiary's residential area in the US |
| `charges` | **(Target)** Annual medical insurance cost billed |

### 2. Exploratory Data Analysis (EDA)

- Confirmed the dataset is complete (1,338 non-null rows across every column, no duplicates)
- **`age`** — a high concentration of individuals around 20, with a relatively uniform spread across the remaining adult age brackets up to 60
- **`sex`** — a fairly balanced split, with slightly more males than females
- **`bmi`** — an approximately normal, bell-shaped distribution centered around ~30 (the clinically "normal" range is 18.5–24.9, so the average beneficiary here trends overweight)
- **`children`** — right-skewed, with most individuals having zero children and frequency decreasing as the count increases
- **`smoker`** — significant class imbalance, with far more non-smokers than smokers
- **`region`** — a nearly even split across all four US regions, with a slight lean toward the southeast
- **`charges`** (target) — heavily right-skewed with a long tail and a secondary hump, meaning most charges cluster at the lower end while a smaller subset of individuals face much higher costs. The mean charge ($13,270.42) sits well above the median ($9,382.03), confirming this skew

> The notebook's EDA notes that a log transform (`np.log1p`) on `charges` would help normalize this skew; the transform was explored as a recommendation but the modeling stage below was trained on the raw `charges` values.

### 3. Feature Encoding & Scaling

- **Categorical features** (`sex`, `smoker`, `region`) are transformed using **One-Hot Encoding** (`OneHotEncoder`)
- **Numerical features** (`age`, `bmi`, `children`) are **standardized** with `StandardScaler`
- Both are combined via a `ColumnTransformer`, wrapped together with the model in a single `Pipeline` (`make_pipeline`)

### 4. Model Selection

Seven regression algorithms were trained on an 80/20 train-test split and compared on **R²** and **MAE**:

| Model | R² Score | MAE |
|---|---|---|
| Linear Regression | 0.8000 | $3,933.27 |
| Ridge Regression | 0.7996 | $3,945.54 |
| Lasso Regression | 0.7999 | $3,934.34 |
| KNN | 0.7315 | $4,013.76 |
| Decision Tree | 0.6470 | $3,735.20 |
| SVM (RBF) | 0.8771 | $1,838.30 |
| **Random Forest** | **0.8954** | $2,486.53 |

The notebook's evaluation highlights a trade-off: **Random Forest** achieves the highest R², but **SVR has a substantially lower MAE**, meaning its individual predictions tend to be more accurate on average with fewer large errors, even though its R² is slightly lower. The notebook's write-up leans toward recommending SVR for this reason — worth weighing both metrics against your app's priorities. The pipeline that was ultimately validated, checked for overfitting, and serialized to `pipe.pkl` is the **Random Forest** configuration (`n_estimators=100, max_depth=15, max_features=0.75, max_samples=0.5`).

### 5. Model Validation

- **Overfitting check:** Train R² = 0.9383 vs. Test R² = 0.8954 — a moderate gap, indicating some overfitting to the training data but still strong generalization
- **5-fold Cross-Validation:** R² scores per fold were 0.8383, 0.8230, 0.8704, 0.7655, 0.8177, for a **mean CV R² of 0.8230** — reasonably consistent across folds, with Fold 4 showing the most variance

### 6. Deployment Preparation

The final trained pipeline and dataset are serialized with `pickle`:

| File | Contents |
|---|---|
| `pipe.pkl` | Full trained pipeline (One-Hot Encoding + Scaling + Random Forest model) |
| `df.pkl` | Original insurance dataframe, used to populate dropdown/input options (sex, smoker, region values, etc.) in the app |

The Streamlit app loads both files at startup — `df.pkl` to populate the input fields, and `pipe.pkl` to make predictions directly from raw input (no manual encoding needed, since it's baked into the pipeline).

## 🛠️ Tech Stack

- **Python**
- **pandas / numpy** — data processing
- **scikit-learn** — `OneHotEncoder`, `StandardScaler`, `ColumnTransformer`, `Pipeline`, `KFold`/`cross_val_score`, and multiple regressors (Linear, Ridge, Lasso, KNN, Decision Tree, SVM, Random Forest, Extra Trees)
- **matplotlib / seaborn** — exploratory data analysis
- **Streamlit** — web app frontend
- **pickle** — model/data serialization

## 📁 Project Structure

```
medical-insurance-predictor/
├── app.py                              # Streamlit frontend
├── medical-insurance-predictor.ipynb   # EDA, preprocessing & model-building notebook
├── pipe.pkl                            # Serialized trained pipeline (encoder + scaler + Random Forest model)
├── df.pkl                              # Serialized insurance dataframe (for input options)
├── dataset/                            # Raw dataset (insurance.csv)
├── Results/                            # Screenshots of app results
├── requirements.txt                    # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Donna152/medical-insurance-predictor.git
   cd medical-insurance-predictor
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure `pipe.pkl` and `df.pkl` are present in the project root. If not, run through `medical-insurance-predictor.ipynb` to regenerate them from `dataset/insurance.csv`.

4. Run the app locally:
   ```bash
   streamlit run app.py
   ```

5. Open the URL shown in your terminal (typically `http://localhost:8501`).

## 🌐 Deployment

This app is built to be deployed on **Streamlit Community Cloud**.

To deploy your own copy:
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Select the repository, branch, and `app.py` as the entry point.
4. Deploy — Streamlit Cloud will install dependencies from `requirements.txt` automatically.

## 📓 Notebook

`medical-insurance-predictor.ipynb` contains the full, step-by-step pipeline used to build this system:
- Dataset loading and completeness checks (no missing values, no duplicates)
- Exploratory data analysis across all 6 predictor variables and the `charges` target
- Feature/target split and train-test split
- One-Hot Encoding and feature scaling via `ColumnTransformer` and `StandardScaler`
- Training and comparison of 7 regression algorithms
- Overfitting check (train vs. test R²) and 5-fold cross-validation
- Serialization of the final `pipe.pkl` and `df.pkl` used by the Streamlit app

## 🔮 Possible Improvements

- Apply the log transform (`np.log1p`) on `charges` that the EDA recommended, then compare model performance against the raw-target results above
- Address the Random Forest/SVR trade-off directly — e.g. try SVR hyperparameter tuning or a Voting/Stacking ensemble combining both strengths
- Investigate Fold 4's lower cross-validation score to understand which data subset the model handles less reliably
- Add interaction features (e.g. `smoker` × `bmi`), since smoking status combined with high BMI is a well-known cost driver in insurance data

## 📄 License

This project is intended for educational and portfolio purposes. The dataset used is the [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance), publicly available on Kaggle.
