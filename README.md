# ML Classification Pipeline

This is a  code repository  for the paper ... .Here we implements a machine learning classification pipeline for binary and multi-class tasks using various algorithms. It supports both training (with and without hyperparameter tuning), model evaluation using confusion matrices and ROC-AUC, and automatic saving of results.



## 🔧 Features

- Supports multiple classifiers: SVM, Random Forest, Logistic Regression, XGBoost, Decision Tree, KNN
- Optional hyperparameter tuning using GridSearchCV
- Saves trained models, confusion matrix plots, and evaluation metrics
- Modular codebase for easy customization and extension

## ▶️ How to Run

1. Ensure your dataset is located in the `Data/` folder.
2. Modify `main_11_binary.py` to use the correct dataset filename and target column (e,g attack type binary) if needed.
3. Set tuning = True to perform hyperparameter search and model optimization.
4. Run the script:

```bash
git clone https://github.com/your-username/ml-classification-pipeline.git
cd ml-classification-pipeline
pip install -r requirements.txt
python main_11_binary.py

---
