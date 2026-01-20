"""
Ensemble Model Evaluation for Rice Disease Detection
Combines predictions from multiple models for improved accuracy
"""

import os
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import config

def find_best_model(model_type):
    """Find the best model file for a given model type"""
    models_dir = config.MODELS_DIR
    model_files = [f for f in os.listdir(models_dir) if f.startswith(model_type) and f.endswith('_best.keras')]
    
    if not model_files:
        print(f"⚠️  No model found for {model_type}")
        return None
    
    # Sort by date and get latest
    model_files.sort(reverse=True)
    model_path = os.path.join(models_dir, model_files[0])
    print(f"✅ Found {model_type}: {model_files[0]}")
    return model_path

def load_models():
    """Load all available trained models"""
    models = {}
    
    # Try to load each model
    for model_type in ['VGG16', 'ResNet50', 'EfficientNetB0']:
        model_path = find_best_model(model_type)
        if model_path and os.path.exists(model_path):
            try:
                models[model_type] = keras.models.load_model(model_path)
                print(f"✅ Loaded {model_type}")
            except Exception as e:
                print(f"❌ Failed to load {model_type}: {e}")
    
    return models

def create_validation_generator():
    """Create validation data generator"""
    val_datagen = ImageDataGenerator(rescale=config.RESCALE)
    
    val_generator = val_datagen.flow_from_directory(
        config.VALIDATION_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        classes=config.CLASS_NAMES,
        shuffle=False,
        seed=config.RANDOM_SEED
    )
    
    return val_generator

def ensemble_predict(models, val_generator, weights=None):
    """
    Make ensemble predictions
    
    Args:
        models: Dictionary of loaded models
        val_generator: Validation data generator
        weights: Optional model weights (dict with model_type: weight)
    
    Returns:
        Combined predictions, true labels
    """
    print("\n" + "="*70)
    print("ENSEMBLE PREDICTION")
    print("="*70)
    
    # Get true labels
    val_generator.reset()
    y_true = val_generator.classes
    
    # Collect predictions from each model
    all_predictions = {}
    
    for model_name, model in models.items():
        print(f"\n🔮 Predicting with {model_name}...")
        val_generator.reset()
        predictions = model.predict(val_generator, verbose=1)
        all_predictions[model_name] = predictions
        
        # Individual accuracy
        y_pred = np.argmax(predictions, axis=1)
        acc = accuracy_score(y_true, y_pred)
        print(f"   {model_name} individual accuracy: {acc:.4f}")
    
    # Combine predictions with weights
    if weights is None:
        # Equal weighting
        weights = {name: 1.0 / len(models) for name in models.keys()}
    else:
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
    
    print(f"\n📊 Ensemble weights:")
    for name, weight in weights.items():
        print(f"   {name}: {weight:.2%}")
    
    # Weighted average of predictions
    ensemble_predictions = np.zeros_like(list(all_predictions.values())[0])
    for model_name, predictions in all_predictions.items():
        ensemble_predictions += predictions * weights[model_name]
    
    return ensemble_predictions, y_true

def evaluate_ensemble(ensemble_predictions, y_true):
    """Evaluate ensemble predictions"""
    y_pred = np.argmax(ensemble_predictions, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*70)
    print("ENSEMBLE RESULTS")
    print("="*70)
    print(f"✅ Ensemble Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*70)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=4)
    print("\nClassification Report:")
    print("-" * 70)
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.title(f'Ensemble Model - Confusion Matrix\nAccuracy: {accuracy:.2%}',
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    cm_path = os.path.join(config.RESULTS_DIR, 'ensemble_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Confusion matrix saved to: {cm_path}")
    plt.show()
    
    # Save report
    report_path = os.path.join(config.RESULTS_DIR, 'ensemble_report.txt')
    with open(report_path, 'w') as f:
        f.write("ENSEMBLE MODEL EVALUATION\n")
        f.write("="*70 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write(report)
    
    print(f"✅ Report saved to: {report_path}")
    
    return accuracy

def main():
    """Main ensemble evaluation function"""
    print("\n" + "="*70)
    print("RICE DISEASE DETECTION - ENSEMBLE EVALUATION")
    print("="*70 + "\n")
    
    # Load models
    print("Loading trained models...")
    models = load_models()
    
    if not models:
        print("\n❌ No models found! Please train models first.")
        return
    
    print(f"\n✅ Loaded {len(models)} model(s): {list(models.keys())}")
    
    # Create validation generator
    print("\nLoading validation data...")
    val_generator = create_validation_generator()
    print(f"✅ Validation samples: {val_generator.samples}")
    
    # Strategy 1: Equal weighting
    print("\n" + "="*70)
    print("STRATEGY 1: Equal Weighting")
    print("="*70)
    ensemble_pred_equal, y_true = ensemble_predict(models, val_generator)
    acc_equal = evaluate_ensemble(ensemble_pred_equal, y_true)
    
    # Strategy 2: Weighted (if multiple models)
    if len(models) > 1:
        print("\n" + "="*70)
        print("STRATEGY 2: Performance-Based Weighting")
        print("="*70)
        
        # Weight models by their known performance
        # VGG16: 50.6%, ResNet50: 38%, EfficientNet: 25%
        weights = {
            'VGG16': 0.70,  # 70% weight
            'ResNet50': 0.20,  # 20% weight
            'EfficientNetB0': 0.10  # 10% weight
        }
        
        # Only use weights for models that exist
        weights = {k: v for k, v in weights.items() if k in models}
        
        ensemble_pred_weighted, _ = ensemble_predict(models, val_generator, weights=weights)
        acc_weighted = evaluate_ensemble(ensemble_pred_weighted, y_true)
        
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        print(f"Equal weighting:        {acc_equal:.4f} ({acc_equal*100:.2f}%)")
        print(f"Performance weighting:  {acc_weighted:.4f} ({acc_weighted*100:.2f}%)")
        print(f"Best individual (VGG16): 0.5061 (50.61%)")
        print("="*70)
        
        improvement = max(acc_equal, acc_weighted) - 0.5061
        print(f"\n🎉 Ensemble improvement: +{improvement:.4f} (+{improvement*100:.2f}%)")
    
    print("\n✅ Ensemble evaluation complete!")
    print(f"📁 Results saved to: {config.RESULTS_DIR}")

if __name__ == "__main__":
    main()
