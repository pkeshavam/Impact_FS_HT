import os

def save_model(model, file_path):
    """Save the trained model."""
    import joblib
    joblib.dump(model, file_path)

def load_model(file_path):
    """Load a saved model."""
    import joblib
    return joblib.load(file_path)

def ensure_dir(directory):
    """Ensure a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
