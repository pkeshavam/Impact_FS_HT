# ML Classification Pipeline

This is a code repository for our study on the impact of Feature Selection and Hyperparameter Tuning for IIoT Attack Detection.
Using an ensemble of feature selection algorithms, we identify the most relevant features in the Edge-IIoT dataset. Based on the ranking of features thus obtained, we create three distinct datasets. 
We then test these three datasets for three classification types - binary, multigroup and multiclass. 
We use six Machine Learning algorithms for evaluating our datasets with and without hyperparameter tuning. 


## 🔧 Features

- Supports multiple classifiers: LSVC, Random Forest, Logistic Regression, XGBoost, Decision Tree, KNN
- Optional hyperparameter tuning using GridSearchCV
- Saves trained models, confusion matrix plots, and evaluation metrics
- Modular codebase for easy customization and extension

## ▶️ How to Run

1. Ensure your dataset is located in the `Data/` folder.
2. Modify `main_11_binary.py` to use the correct dataset filename and target column (e,g attack type binary) if needed.
3. Set tuning = True to perform hyperparameter search and model optimization.
4. Run the script:

```bash
Download Repository
cd ml-classification-pipeline
pip install -r requirements.txt
python main_11_binary.py

---
