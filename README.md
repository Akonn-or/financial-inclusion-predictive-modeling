
# Financial Inclusion Prediction

This project predicts whether a person has access to a **bank account** using demographic and socioeconomic data. The work focuses on understanding key factors driving financial inclusion and applying supervised machine learning models to classify individuals.

## Dataset

* **Train set**: 23,524 rows × 13 columns
* **Test set**: 10,086 rows × 12 columns
* Features include:

  * Country, Year
  * Location type (Urban/Rural)
  * Cellphone access
  * Household size
  * Age, Gender, Marital status
  * Education level
  * Job type
* **Target**: `bank_account` (Yes/No → 1/0)

## Approach

1. **Data Cleaning & Preprocessing**

   * Handled missing values (bank account label missing in test).
   * Reduced categorical levels (e.g., job types, marital status, education levels).
   * Feature engineering:

     * Age binning (`young`, `adult`, `middle age`, `old age`)
     * Household size binning (`1–4`, `5+`)
     * Relationship grouping (`head`, `spouse`, `child`, `other`)

2. **Exploratory Data Analysis (EDA)**

   * Visualized distributions of age, household size, job type, and education.
   * Explored class imbalance (`No` >> `Yes`).

3. **Modeling**

   * Baseline classifiers:

     * Logistic Regression
     * Decision Tree
     * Random Forest
     * LightGBM
   * Used **SMOTE** and **RandomOverSampler** to handle class imbalance.
   * Encoded categorical variables with **OneHotEncoder**.

4. **Model Tuning**

   * Hyperparameter optimization with **Optuna (Bayesian Search)**.
   * Evaluated using **Mean Absolute Error (MAE)**.

## Results

| Model               | MAE ↓     | Notes                   |
| ------------------- | --------- | ----------------------- |
| Logistic Regression | 0.217     | Weak on imbalanced data |
| Decision Tree       | 0.152     | Simple, interpretable   |
| Random Forest       | 0.154     | Strong baseline         |
| LightGBM            | **0.118** | Best performing         |

* Final chosen model: **LightGBM**
* Validation MAE: ~ **0.12**

##  How to Run


1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

2. Run Jupyter Notebook

   ```bash
   jupyter notebook fin_inclusion_nb.ipynb
   ```

3. Trained model will be saved in:

   ```
   output/model.pkl
   ```

##  File Structure

```
├── data/
│   ├── Train.csv
│   ├── Test.csv
├── output/
│   ├── model.pkl
├── fin_inclusion_nb.ipynb   # Main notebook
├── README.md
└── requirements.txt
```

##  Tech Stack

* Python (Pandas, NumPy, Matplotlib, Seaborn)
* Scikit-learn
* LightGBM
* Optuna (Bayesian optimization)
* Imbalanced-learn (SMOTE, RandomOverSampler)

