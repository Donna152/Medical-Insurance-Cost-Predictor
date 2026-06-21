# Medical Insurance Cost Predictor

A machine learning regression project that predicts individual medical insurance charges based on personal and demographic attributes. The project covers the full ML pipeline — exploratory data analysis, preprocessing, model comparison across five regression algorithms, feature selection, overfitting checks, and cross-validation.

---

## Repository Structure

```
medical-insurance-cost-predictor/
├── data/
│   └── insurance.csv            # Keep your dataset here
├── src/
│   └── medical-insurance-cost-predictor.ipynb         # Keep your Python code here
├── .gitignore                   # Tell Git to ignore the 'data/' folder
└── README.md                    # Your project documentation
```

---

## Dataset Description

The dataset contains information about health insurance beneficiaries used to predict individual medical costs (`charges`).

| Column | Description |
|---|---|
| `age` | Age of the primary beneficiary |
| `sex` | Insurance contractor gender (female, male) |
| `bmi` | Body Mass Index (kg/m²) |
| `children` | Number of children covered / number of dependents |
| `smoker` | Smoking status (yes/no) |
| `region` | Residential area in the US (northeast, southeast, southwest, northwest) |
| `charges` | **Target** — individual medical costs billed by health insurance |

**Variable types:**
- **Numerical:** `age`, `bmi`, `children`
- **Categorical:** `sex`, `smoker`, `region` — require encoding before modeling
- **Target:** `charges` is continuous, making this a **regression task**

---

## Exploratory Data Analysis

### Data Quality
- **1338 rows, 7 columns** — no missing values across any column. No imputation was required.
- 4 numerical columns (`age`, `bmi`, `children`, `charges`) and 3 categorical columns (`sex`, `smoker`, `region`) requiring encoding.

### Summary Statistics
- **Charges skewness:** Mean ($13,270.42) is well above the median ($9,382.03), with a high standard deviation ($12,110.01) — a clear sign of a right-skewed distribution driven by a subset of very high-cost claims.
- **Demographics:** Beneficiaries range from 18 to 64 years old, averaging around 39.
- **BMI:** Average of 30.66, which falls within the clinical "obese" category (>30), suggesting a likely strong correlation with charges.
- **Dependents:** Most beneficiaries have 0–2 children, averaging about 1.

### Outlier Analysis (Boxplots)
- **Children:** Naturally bounded (0–5), no outliers, median of 1.
- **Age:** Well-spread across the full range, no extreme outliers.
- **BMI:** Clear outliers above ~47 — a segment of the population is significantly heavier than the rest.
- **Charges:** Heavy outlier presence beyond $35,000, reaching $60,000+ — confirms strong right-skew driven by high-cost cases.

### Distribution Analysis (Histograms)
- **children:** Right-skewed — most families have 0–1 children.
- **age:** Roughly uniform — good coverage across all adult age groups.
- **bmi:** Approximately normal, centered around 30 — consistent with the obese-average finding.
- **charges:** Highly right-skewed — most charges cluster below $15,000 with a long tail of expensive claims.

### Pairplot Analysis (colored by smoking status)
- **Smoking is the primary cost driver** — smokers show a distinct, consistently higher band of charges compared to non-smokers.
- **Smoker–BMI interaction** — among smokers, higher BMI strongly correlates with higher charges; this relationship is much weaker for non-smokers.
- **Age–cost trend** — three visible cost tiers emerge, largely separated by smoking status, with an overall upward trend in charges as age increases.
- Other features (`age`, `bmi`, `children`) are well-distributed across both smoking groups, reinforcing that **smoking status is the dominant factor** behind cost differences rather than demographic imbalance.

---

## Data Preparation

### Train/Test Split
- 80/20 split using `train_test_split` with `random_state=0`.
- Train set: 1070 samples. Test set: 268 samples.

### Categorical Encoding
- `OneHotEncoder(drop='first')` applied to `sex`, `smoker`, `region`.
- Fit only on `X_train` and applied to `X_test` using `.transform()` only — preventing data leakage from the test set into the encoding scheme.

### Numerical Feature Scaling
- `StandardScaler` applied to `age`, `bmi`, `children`.
- Scaler statistics (mean, std) learned only from `X_train`, then applied to `X_test` — again avoiding leakage.

---

## Models Trained

Five regression models were trained and evaluated on the same 80/20 split:

| Model | MAE | R² Score |
|---|---|---|
| Linear Regression | 3933.27 | 0.8000 |
| Decision Tree Regressor | 3370.17 | 0.6884 |
| Random Forest Regressor | 2614.43 | 0.8816 |
| **Gradient Boosting Regressor** | **2456.03** | **0.8962** |
| XGBoost Regressor | 2607.29 | 0.8824 |

### Model Selection
**Gradient Boosting Regressor** was selected as the final model — it achieved the lowest MAE and the highest R² among all five candidates, offering the best balance of predictive accuracy and computational efficiency.

**Observations on model comparison:**
- The single **Decision Tree** performed worst (R² = 0.6884) — a clear sign of overfitting to training noise without the variance reduction that ensembles provide.
- All three ensemble methods (Random Forest, Gradient Boosting, XGBoost) significantly outperformed both linear regression and the single decision tree, confirming that the relationship between features and charges is **non-linear** and benefits from tree-based ensemble learning.
- Gradient Boosting edged out XGBoost and Random Forest by a small margin, making it the most effective model for this dataset size and feature set.

---

## Overfitting Check (Full Feature Set)

| Metric | Score |
|---|---|
| Train R² | 0.8974 |
| Test R² | 0.8962 |

The very small gap between train and test R² (0.0012) indicates the Gradient Boosting model generalizes well and is **not overfitting** despite using the full feature set.

---

## Feature Selection

Feature importance was extracted directly from the trained Gradient Boosting model.

**Key finding:** Smoking status is the most significant determinant of medical charges by a wide margin, followed by BMI and age. Features like region and sex contribute almost no predictive value.

### Reduced Model
The model was retrained using only the three most important features: `smoker_yes`, `bmi`, `age`.

| Metric | Full Model | Reduced Model |
|---|---|---|
| MAE | 2456.03 | 2382.82 |
| R² Score | 0.8962 | 0.8908 |

**Observation:** Reducing to just 3 features actually **improved MAE** while keeping R² nearly identical. This confirms that the dropped features (`sex`, `region`, `children`) were contributing mostly noise rather than useful signal, allowing for a simpler, more interpretable, and equally accurate model.

### Overfitting Check (Reduced Model)

| Metric | Score |
|---|---|
| Train R² | 0.8905 |
| Test R² | 0.8908 |

The test R² being marginally higher than the train R² is a strong indicator that the reduced model generalizes extremely well, with no signs of overfitting.

---

## K-Fold Cross-Validation

5-fold cross-validation was performed on the reduced model to obtain a more robust performance estimate.

| Fold | R² Score |
|---|---|
| 1 | 0.8573 |
| 2 | 0.8076 |
| 3 | 0.8355 |
| 4 | 0.8691 |
| 5 | 0.8783 |
| **Mean** | **0.8496** |

**Observation:** The single train-test split gave an optimistic R² of 0.8908, while the more rigorous 5-fold cross-validation gives a more conservative and realistic mean R² of 0.8496. The fold scores range from 0.8076 to 0.8783 — some natural variance across data subsets, but the model consistently maintains strong predictive performance, confirming the chosen 3-feature model is **stable** and not the result of a lucky train-test split.

---

## Results and Conclusion

1. **Feature impact:** Feature selection revealed that `smoker`, `bmi`, and `age` are the true drivers of medical costs, while `region` and `sex` contributed mostly noise. Removing them produced a cleaner, more interpretable model without sacrificing accuracy.

2. **Overfitting check:** The reduced model achieved a test R² of 0.8908, remarkably close to its training R² of 0.8905. This minimal gap confirms the model generalizes well to unseen data.

3. **Cross-validation honesty:** While the single train-test split gave R² = 0.8908, the 5-fold cross-validation gave a more conservative mean R² of 0.8496 — a more reliable estimate of real-world performance.

4. **Stability:** Individual fold scores ranged from 0.8076 to 0.8783. This fluctuation is expected and reasonable, and the model consistently performs well across different data subsets.

5. **Final verdict:** The reduced-feature Gradient Boosting model is the most effective solution for this project — it balances high predictive accuracy, reduced complexity (3 features instead of 8), and proven stability through cross-validation, making it well-suited for deployment.

---

## Getting Started

### Requirements
```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
```

### Running the Notebook

1. Clone the repository:
```bash
git clone https://github.com/your-username/medical-insurance-cost-predictor.git
cd medical-insurance-cost-predictor
```

2. Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

3. Make sure `insurance.csv` is in the same directory as the notebook, then open and run:
```bash
jupyter notebook medical-insurance-cost-predictor.ipynb
```

---

## Key Takeaways

- **Domain insight beats feature count.** A 3-feature model (smoker, bmi, age) performs essentially as well as the full 8-feature model — proving that understanding *which* features matter is more valuable than including everything.
- **Smoking status dominates the prediction.** It is by far the single strongest predictor of medical charges in this dataset.
- **Cross-validation matters.** A single train-test split can be optimistic; k-fold cross-validation gave a more honest, slightly lower performance estimate that should be trusted over the single split result.
- **Tree-based ensembles outperform linear models** on this dataset, indicating non-linear relationships between features (especially the smoker × bmi interaction) and the target.

---

## License

This project is released for academic and educational use. Feel free to use, modify, and share with attribution.
