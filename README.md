# Credit Scoring Classification Model 📊💳

## Objective
The objective of this project is to predict an individual's creditworthiness and assess the risk of loan default using historical financial data. This classification model helps financial institutions make data-driven lending decisions.

## Dataset
This project uses the **Credit Risk Dataset** from Kaggle, which includes real-world features such as:
- Person Age & Income
- Home Ownership Status
- Employment Length
- Loan Intent & Loan Amount
- Interest Rates & Historical Default Records

## Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Random Forest Classifier)
- **Data Preprocessing:** StandardScaler, One-Hot Encoding (`pd.get_dummies`)

## Key Features & Workflow
1. **Data Cleaning:** Handled missing values in continuous variables (e.g., employment length, interest rates) using median imputation.
2. **Feature Engineering:** Converted categorical variables into numeric formats using one-hot encoding.
3. **Data Scaling:** Applied `StandardScaler` to normalize financial figures, ensuring unbiased model weight distribution.
4. **Model Training:** Trained a robust **Random Forest Classifier** to capture complex, non-linear relationships in the financial data.
5. **Evaluation:** Assessed model accuracy using comprehensive metrics:
   - Precision & Recall
   - F1-Score
   - ROC-AUC Score

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/Credit-Scoring-Model.git](https://github.com/your-username/Credit-Scoring-Model.git)
   cd Credit-Scoring-Model
