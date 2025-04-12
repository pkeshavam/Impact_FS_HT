import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_dir(directory):
    """
    Ensure the specified directory exists; create it if it doesn't.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_model(file_path):
    """
    Load a saved model from the specified path.
    """
    return joblib.load(file_path)

def plot_confusion_matrix(y_true, y_pred, current_CSV, model_name, tuned, selected_model, save_dir="Plots"):
    print(f"plot_confusion_matrix: tuned={tuned}")
    print(f"{model_name}")
    print(f"{selected_model}")
    
    # Determine number of features
    if "11" in current_CSV:
        num_classes = "11"
    elif "25" in current_CSV:
        num_classes = "25"
    elif "35" in current_CSV:
        num_classes = "35"
    else:
        num_classes = "Unknown"
        
    # Determine the colormap and size based on classification
    if "binary" in current_CSV.lower():
        cmap = "Reds"
        plt.figure(figsize=(4, 4))
        font_size = 14  # Increased font size for binary matrix
    elif "multigroup" in current_CSV.lower():
        cmap = "Greens"
        plt.figure(figsize=(4, 4))
        font_size = 10  # Default font size
    elif "multiclass" in current_CSV.lower():
        cmap = "Blues"
        plt.figure(figsize=(8, 6))
        font_size = 10  # Default font size
    else:
        cmap = "Blues"  # Default colormap if none of the conditions are met
        plt.figure(figsize=(8, 6))
        font_size = 10  # Default font size

    ensure_dir(save_dir)
    conf_matrix = confusion_matrix(y_true, y_pred)

    ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=cmap, cbar=False, annot_kws={"size": font_size})  # Added font size variable
    
    # Adjust tick positions to be centered between grid lines
    tick_positions = [i + 0.5 for i in range(len(conf_matrix))]  # Center ticks between lines
    
    ax.set_xticks(tick_positions)  # Set x-tick positions
    ax.set_yticks(tick_positions)  # Set y-tick positions

    ax.set_xticklabels(range(len(conf_matrix)))  # Set x-tick labels
    ax.set_yticklabels(range(len(conf_matrix)))  # Set y-tick labels

    # Handle x and y ticks based on conditions
    if not tuned:  # No tuning cases
        if num_classes in ["11", "25"]:
            plt.ylabel(f"Actual")
            plt.xlabel("")  # No x-label for these cases
            ax.set_xticks([])   # Remove x-ticks for these cases
        elif num_classes == "35":
            plt.ylabel(f"Actual")
            plt.xlabel(f"Predicted")
    else:  # Tuning cases
        if num_classes in ["11", "25"]:
            ax.set_xticks([])   # Remove x-ticks for these cases
            ax.set_yticks([])   # Remove y-ticks for these cases
            plt.xlabel("")      # No x-label for these cases
            plt.ylabel("")      # No y-label for these cases
        elif num_classes == "35":
            plt.xlabel(f"Predicted")
            plt.ylabel("")      # No y-label for this case
            ax.set_yticks([])   # Remove y-ticks for these cases
    
    #plt.title(f"{model_name} ({'with_tuning' if tuned else 'no_tuning'})")
    plt.title(f"{selected_model} {num_classes} Features ({'with tuning' if tuned else 'no tuning'})")
    #plt.xlabel("Predicted")
    #plt.ylabel("Actual")

    filename = f"{model_name.lower().replace(' ', '_')}_{'with_tuning' if tuned else 'no_tuning'}_confusion_matrix.png"
    print(f"{filename}")
    #plt.savefig(os.path.join(save_dir, filename))
    # Save figure with tight bounding box and no padding
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {os.path.join(save_dir, filename)}.")

def evaluate_roc_auc(y_true, y_prob, n_classes):
    """
    Compute ROC-AUC score for multi-class classification using one-vs-rest strategy.
    """
    roc_auc = {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
        roc_auc[i] = roc_auc_score(y_true == i, y_prob[:, i])
    return roc_auc

def evaluate(directory, X_test, y_test, current_CSV, tuning, selected_model, save_metrics_to, save_dir="Plots"):
    """
    Load models from a directory, evaluate them, and save metrics and plots.
    """
    print(f"Evaluating models in directory: {directory}...")
    ensure_dir(save_dir)

    metrics = {}
    n_classes = len(set(y_test))  # Determine the number of 
    
    # Replace spaces with underscores in selected_model
    selected_model_underscore = selected_model.replace(" ", "_")
    
    # Find the specific file
    target_file = None
    for file in os.listdir(directory):
        if file.endswith(".pkl") and current_CSV in file and selected_model_underscore.lower() in file.lower():
            target_file = file
            break

    if target_file is None:
        print(f"No matching file found for {selected_model} and {current_CSV}")
        return

    model_path = os.path.join(directory, target_file)
    model_name = target_file.replace(".pkl", "").replace("_", " ").title()
    print(f"file: {target_file}")
    print(f"model_name: {model_name}")
    tuned = tuning
    print(f"evaluate: tuned={tuned}")
    
    # Load model
    model = load_model(model_path)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Compute metrics
    model_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_score_macro": f1_score(y_test, y_pred, average="macro"),
    }
    if y_pred_prob is not None:
        roc_auc = evaluate_roc_auc(y_test, y_pred_prob, n_classes)
        model_metrics.update({f"roc_auc_class_{i}": roc_auc[i] for i in range(n_classes)})

    metrics[model_name] = model_metrics

    # Save plots
    plot_confusion_matrix(y_test, y_pred, current_CSV, model_name, tuned, selected_model, save_dir)
    #if y_pred_prob is not None:
    #    for i in range(n_classes):
    #        plt.figure()
    #        fpr, tpr, _ = roc_curve(y_test == i, y_pred_prob[:, i])
    #        plt.plot(fpr, tpr, label=f"Class {i} ROC Curve")
    #        plt.xlabel("False Positive Rate")
    #        plt.ylabel("True Positive Rate")
    #        plt.title(f"ROC Curve for Class {i} - {model_name}")
    #        plt.legend()
    #        filename = f"{model_name.lower().replace(' ', '_')}_class_{i}_roc_curve.png"
    #        plt.savefig(os.path.join(save_dir, filename))
    #        plt.close()
    #        print(f"ROC curve for class {i} saved to {os.path.join(save_dir, filename)}.")
                    
    # Save metrics to a CSV file
    metrics_df = pd.DataFrame(metrics).T  # Transpose for readability
    metrics_df.to_csv(save_metrics_to)
    print(f"Metrics saved to {save_metrics_to}.")

