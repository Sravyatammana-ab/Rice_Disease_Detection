"""
Evaluate model performance on rice disease detection
Generates confusion matrix, classification report, ROC curves, and other metrics
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import pandas as pd

import tensorflow as tf
from tensorflow import keras

import config
from data_loader import RiceDiseaseDataLoader


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    # plt.show()  # Disabled - auto-save only
    
    # Calculate and print per-class accuracy
    print("\nPer-class Accuracy:")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        class_acc = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        print(f"  {class_name}: {class_acc:.4f} ({cm[i, i]}/{cm[i, :].sum()})")
    
    return cm


def plot_roc_curves(y_true, y_pred_proba, class_names, save_path=None):
    """
    Plot ROC curves for each class
    
    Args:
        y_true: True labels (integers)
        y_pred_proba: Predicted probabilities (one-hot or probabilities)
        class_names: List of class names
        save_path: Path to save the plot
    """
    n_classes = len(class_names)
    
    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.4f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-class', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curves saved to: {save_path}")
    
    # plt.show()  # Disabled - auto-save only
    
    # Print AUC scores
    print("\nAUC Scores:")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {roc_auc[i]:.4f}")
    print(f"  Mean AUC: {np.mean(list(roc_auc.values())):.4f}")
    
    return roc_auc


def plot_prediction_samples(model, generator, class_names, num_samples=16, save_path=None):
    """
    Plot sample predictions with true and predicted labels
    
    Args:
        model: Trained model
        generator: Data generator
        class_names: List of class names
        num_samples: Number of samples to display
        save_path: Path to save the plot
    """
    # Get a batch of images
    generator.reset()
    images, labels = next(generator)
    
    # Make predictions
    predictions = model.predict(images[:num_samples])
    
    # Calculate grid size
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        # Get true and predicted labels
        true_idx = np.argmax(labels[i])
        pred_idx = np.argmax(predictions[i])
        true_label = class_names[true_idx]
        pred_label = class_names[pred_idx]
        confidence = predictions[i][pred_idx]
        
        # Display image
        axes[i].imshow(images[i])
        
        # Set title with color based on correctness
        if true_idx == pred_idx:
            color = 'green'
            title = f'✓ {pred_label}\n({confidence:.2%})'
        else:
            color = 'red'
            title = f'✗ Pred: {pred_label} ({confidence:.2%})\nTrue: {true_label}'
        
        axes[i].set_title(title, fontsize=10, fontweight='bold', color=color)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Prediction samples saved to: {save_path}")
    
    # plt.show()  # Disabled - auto-save only


def evaluate_model(model_path, use_validation=True):
    """
    Evaluate a trained model
    
    Args:
        model_path: Path to saved model
        use_validation: Whether to use validation set (True) or test set (False)
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*70)
    print("RICE DISEASE DETECTION - MODEL EVALUATION")
    print("="*70 + "\n")
    
    # Load model
    print(f"[1/5] Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully")
    
    # Load data
    print(f"\n[2/5] Loading dataset...")
    data_loader = RiceDiseaseDataLoader()
    
    if use_validation:
        generator = data_loader.create_validation_generator()
        dataset_name = "Validation"
    else:
        # For test set, you'd create a test generator
        generator = data_loader.create_validation_generator()
        dataset_name = "Test"
    
    print(f"✓ {dataset_name} dataset loaded: {generator.samples} samples")
    
    # Make predictions
    print(f"\n[3/5] Making predictions...")
    generator.reset()
    y_true = generator.classes
    y_pred_proba = model.predict(generator, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    print(f"✓ Predictions completed")
    
    # Calculate metrics
    print(f"\n[4/5] Calculating metrics...")
    
    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)
    print(f"\n{'='*70}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*70}")
    
    # Classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=config.CLASS_NAMES,
        digits=4
    )
    print("\nClassification Report:")
    print("-" * 70)
    print(report)
    
    # Save classification report
    report_path = os.path.join(config.RESULTS_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    print(f"✓ Classification report saved to: {report_path}")
    
    # Visualizations
    print(f"\n[5/5] Creating visualizations...")
    
    # Confusion Matrix
    cm_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')
    cm = plot_confusion_matrix(y_true, y_pred, config.CLASS_NAMES, save_path=cm_path)
    
    # ROC Curves
    roc_path = os.path.join(config.RESULTS_DIR, 'roc_curves.png')
    roc_auc = plot_roc_curves(y_true, y_pred_proba, config.CLASS_NAMES, save_path=roc_path)
    
    # Sample predictions
    samples_path = os.path.join(config.RESULTS_DIR, 'prediction_samples.png')
    plot_prediction_samples(model, generator, config.CLASS_NAMES, 
                           num_samples=16, save_path=samples_path)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETED!")
    print("="*70)
    
    # Return metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'classification_report': report
    }
    
    return metrics


def compare_models(model_paths):
    """
    Compare multiple models
    
    Args:
        model_paths: List of paths to saved models
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70 + "\n")
    
    results = []
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"\nEvaluating: {model_name}")
        print("-" * 70)
        
        # Load and evaluate model
        model = keras.models.load_model(model_path)
        data_loader = RiceDiseaseDataLoader()
        generator = data_loader.create_validation_generator()
        
        # Predictions
        generator.reset()
        y_true = generator.classes
        y_pred_proba = model.predict(generator, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(y_true == y_pred)
        
        results.append({
            'Model': model_name,
            'Accuracy': accuracy
        })
        
        print(f"Accuracy: {accuracy:.4f}")
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('Accuracy', ascending=False)
    
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70 + "\n")
    
    # Save comparison
    comparison_path = os.path.join(config.RESULTS_DIR, 'model_comparison.csv')
    df.to_csv(comparison_path, index=False)
    print(f"✓ Comparison saved to: {comparison_path}")
    
    return df


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description='Evaluate Rice Disease Detection Model')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to saved model file (.h5)'
    )
    parser.add_argument(
        '--compare',
        type=str,
        nargs='+',
        help='Paths to multiple models for comparison'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple models
        compare_models(args.compare)
    else:
        # Evaluate single model
        metrics = evaluate_model(args.model)
        
        print(f"\n✓ Evaluation completed!")
        print(f"✓ Check the '{config.RESULTS_DIR}' folder for detailed results\n")


if __name__ == "__main__":
    main()

