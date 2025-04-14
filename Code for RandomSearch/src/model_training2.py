import os
import time
import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier  # Import RandomForest
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV  # Import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import pandas as pd  # Import for saving training times

def ensure_dir(directory):
    """
    Ensure the specified directory exists; create it if it doesn't.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def train_without_tuning(selected_model, X_train, y_train, models_params):
    """
    Train models without hyperparameter tuning and record training times.
    """
    trained_models = {}
    training_times = {}
    
    # Check if selected_model is a dictionary
    if isinstance(selected_model, dict):
        selected_model = list(selected_model.keys())[0]  # Take the first key if it's a dict
    
    if selected_model not in models_params:
        raise ValueError(f"Model '{selected_model}' not found in models_params.")

    model_class, params = models_params[selected_model]
    print(f"Training {selected_model} with specific parameters...")
    
    # Create model instance with specific parameters
    specific_params = {k: v[0] for k, v in params.items()}
    model = model_class.__class__(**specific_params)
    print(f"Parameters: {specific_params}\n")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()

    training_time = end_time - start_time
    training_times[selected_model] = end_time - start_time
    trained_models[selected_model] = model
    print(f"{selected_model} training completed in {training_time:.2f} seconds.")
    print(f"Parameters used: {specific_params}\n")
    
    return trained_models, training_times, specific_params

def train_with_tuning(models_params, X_train, y_train, n_iter=20):
    """
    Train models with hyperparameter tuning using RandomizedSearchCV and record training times.
    """
    tuned_models = {}
    training_times = {}
    best_hyperparameters = {}
    print(f"Training with tuning starting.")
    for name, (model, param_grid) in models_params.items():
        print(f"Tuning and training {name}...")
        start_time = time.time()
        random_search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=n_iter,  # Number of parameter settings sampled
            scoring=make_scorer(accuracy_score),
            cv=5,
            verbose=1,
            random_state=42  # Set a random state for reproducibility
        )
        random_search.fit(X_train, y_train)
        end_time = time.time()
        training_times[name] = end_time - start_time
        tuned_models[name] = random_search.best_estimator_
        best_hyperparameters[name] = random_search.best_params_
        print(f"{name} training with tuning completed in {training_times[name]:.2f} seconds.")
        print(f"Best parameters for {name}: {random_search.best_params_}\n")
    
    return tuned_models, training_times, best_hyperparameters

def save_models(models, current_CSV, save_dir, training_times=None, save_train_times=True):
    """
    Save trained models to the specified directory and optionally save training times.
    """
    print(f"{current_CSV}")
    
    ensure_dir(save_dir)
    for name, model in models.items():
        file_path = os.path.join(save_dir, f"{name.lower().replace(' ', '_')}_model_{current_CSV}.pkl")
        joblib.dump(model, file_path)
        print(f"Model saved: {file_path}")
    
    print(f"save_train_times={save_train_times}")
    print(f"training_times={training_times}")
    
    if save_train_times and training_times:
        train_times_file = os.path.join(save_dir, "training_times_{current_CSV}.csv")
        pd.DataFrame(training_times, index=["Training Time"]).T.to_csv(train_times_file)
        print(f"Training times saved to: {train_times_file}")
