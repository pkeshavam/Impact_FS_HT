from src.data_processing import load_data, preprocess_data
from src.model_training2 import train_without_tuning, train_with_tuning, save_models
from src.evaluation import evaluate
import os
import pandas as pd
from sklearn.svm import LinearSVC  # Changed SVC to LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def main():
    # Directories and paths
    data_dir = "Data"
    dataset_file = "ML-EdgeIIoT-dataset_25_binary.csv"
    current_CSV = "25_binary"
    target_column = "Attack_label"
    #target_column = "Attack_type_encoded"
    #target_column = "Attack_group_encoded"
    
    models_dir = "Models"
    plots_dir = "Plots"
    metrics_dir = "Metrics"

    # Ensure necessary directories exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Load and preprocess the dataset
    print("Loading and preprocessing data...")
    data = load_data(data_dir, dataset_file)
    X_train, X_test, y_train, y_test = preprocess_data(data, target_column)
    print("Processing finished.")
    print(f"Number of training examples: {X_train.shape[0]}")

    # Define models and hyperparameter grids
    models = {
        "LSVC": LinearSVC(random_state=0),
        "LR": LogisticRegression(random_state=0, max_iter=1000),
        "XGB": XGBClassifier(eval_metric='mlogloss'),
        "KNN": KNeighborsClassifier(),
        "RF": RandomForestClassifier(),
        "DT": DecisionTreeClassifier()
    }

    models_params = {
        "LSVC":  (LinearSVC(random_state=0), {  # Changed SVC to LinearSVC
            "C": [0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "max_iter": [1000],  # Add max_iter since LinearSVC can be sensitive to iterations
            "tol": [1e-4]  # Tolerance for stopping criteria
        }),
        "KNN": (KNeighborsClassifier(), {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
            "metric": ["manhattan", "euclidean", "minkowski"]
        }),
        "RF": (RandomForestClassifier(), {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, None],
            "min_samples_split": [2],
            "min_samples_leaf": [1]
        }),
        "LR": (LogisticRegression(random_state=0, max_iter=1000), {
            "C": [0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["liblinear", "lbfgs","newton-cg", "sag", "saga"]
        }),
       "XGB": (XGBClassifier(eval_metric='mlogloss'), {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "colsample_bytree": [0.8, 1.0]
        }),
        "DT": (DecisionTreeClassifier(), {
            "max_depth": [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        })
    }

    # User inputs
    print("Available models:", list(models.keys()))
    #selected_model = input("Enter the model name to train and test: ").strip()
    #tuning = input("Do you want to run with hyperparameter tuning? (yes/no): ").strip().lower() == "yes"
    
    #selected_model = ("Random Forest").strip()
    
    #Loop through all models without tuning.
    tuning  = False
    # Loop through all models
    for selected_model in models.keys():
        print(f"\nProcessing model: {selected_model}")

        # Set directories based on tuning choice
        tuning_suffix = "with_tuning" if tuning else "no_tuning"
        models_save_dir = os.path.join(models_dir, tuning_suffix)
        plots_save_dir = os.path.join(plots_dir, tuning_suffix)
        metrics_save_dir = metrics_dir
    
        # Train the selected model
        print(f"Training {selected_model} {'with' if tuning else 'without'} hyperparameter tuning...")
        if tuning:
            trained_model, train_times, best_hyperparameters = train_with_tuning(
                {selected_model: models_params[selected_model]}, X_train, y_train
            )
        else:
            trained_model, train_times = train_without_tuning(
                {selected_model: models[selected_model]}, X_train, y_train
            )

        # Save trained model and training time
        save_models(trained_model, current_CSV, save_dir=models_save_dir)
        pd.DataFrame(train_times, index=["Training Time"]).T.to_csv(
            os.path.join(metrics_save_dir, f"{selected_model}_train_time_{tuning_suffix}_{current_CSV}.csv")
        )
        if tuning:
            print(f"best hyperparameters: {best_hyperparameters}")
            # Flatten the dictionary
            flattened_hyperparameters = {k: v for model, params in best_hyperparameters.items() for k, v in params.items()}
        
            # Create DataFrame and save to CSV
            pd.DataFrame(flattened_hyperparameters, index=[selected_model]).to_csv(
                os.path.join(metrics_save_dir, f"{selected_model}_hyperparameters_{tuning_suffix}_{current_CSV}.csv")
            )
        #else:
            #print(f"Parameters Used: {specific_params}")

            # # Create DataFrame and save to CSV
            # pd.DataFrame([specific_params], index=[selected_model]).to_csv(
            #     os.path.join(metrics_save_dir, f"{selected_model}_specific_params_{tuning_suffix}_{current_CSV}.csv")
            # )
            
        saveMetrics=os.path.join(metrics_save_dir, f"{selected_model}_evaluation_metrics_{tuning_suffix}_{current_CSV}.csv")
        print(f"saveMetrics={saveMetrics}")
    
        # Evaluate the selected model
        print(f"Evaluating {selected_model}...")
        evaluate(models_save_dir, X_test, y_test, current_CSV, tuning, selected_model,
                save_metrics_to=saveMetrics,
                save_dir=plots_save_dir)

        print(f"Training and evaluation of {selected_model} completed.")
        
    #Loop through all models with tuning.
    # tuning  = True
    # # Loop through all models
    # for selected_model in models.keys():
    #     print(f"\nProcessing model: {selected_model}")

    #     # Set directories based on tuning choice
    #     tuning_suffix = "with_tuning" if tuning else "no_tuning"
    #     models_save_dir = os.path.join(models_dir, tuning_suffix)
    #     plots_save_dir = os.path.join(plots_dir, tuning_suffix)
    #     metrics_save_dir = metrics_dir
    
    #     # Train the selected model
    #     print(f"Training {selected_model} {'with' if tuning else 'without'} hyperparameter tuning...")
    #     if tuning:
    #         trained_model, train_times, best_hyperparameters = train_with_tuning(
    #             {selected_model: models_params[selected_model]}, X_train, y_train
    #         )
    #     else:
    #         trained_model, train_times, specific_params = train_without_tuning(
    #             {selected_model: models[selected_model]}, X_train, y_train, models_params
    #         )

    #     # Save trained model and training time
    #     save_models(trained_model, current_CSV, save_dir=models_save_dir)
    #     pd.DataFrame(train_times, index=["Training Time"]).T.to_csv(
    #         os.path.join(metrics_save_dir, f"{selected_model}_train_time_{tuning_suffix}_{current_CSV}.csv")
    #     )
    #     if tuning:
    #         print(f"best hyperparameters: {best_hyperparameters}")
    #         # Flatten the dictionary
    #         flattened_hyperparameters = {k: v for model, params in best_hyperparameters.items() for k, v in params.items()}
        
    #         # Create DataFrame and save to CSV
    #         pd.DataFrame(flattened_hyperparameters, index=[selected_model]).to_csv(
    #             os.path.join(metrics_save_dir, f"{selected_model}_hyperparameters_{tuning_suffix}_{current_CSV}.csv")
    #         )
    #     else:
    #         print(f"Parameters Used: {specific_params}")

    #         # Create DataFrame and save to CSV
    #         pd.DataFrame([specific_params], index=[selected_model]).to_csv(
    #             os.path.join(metrics_save_dir, f"{selected_model}_specific_params_{tuning_suffix}_{current_CSV}.csv")
    #         )
            
    #     saveMetrics=os.path.join(metrics_save_dir, f"{selected_model}_evaluation_metrics_{tuning_suffix}_{current_CSV}.csv")
    #     print(f"saveMetrics={saveMetrics}")
    
    #     # Evaluate the selected model
    #     print(f"Evaluating {selected_model}...")
    #     evaluate(models_save_dir, X_test, y_test, current_CSV, tuning, selected_model,
    #             save_metrics_to=saveMetrics,
    #             save_dir=plots_save_dir)

    #     print(f"Training and evaluation of {selected_model} completed.")

if __name__ == "__main__":
    main()
