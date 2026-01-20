"""
Binary Classification Evaluation - Healthy vs Diseased
Simplifies 4-class problem to 2-class problem for higher accuracy
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
        return None
    
    model_files.sort(reverse=True)
    model_path = os.path.join(models_dir, model_files[0])
    return model_path

def load_model_for_binary():
    """Load the best performing model (VGG16)"""
    model_path = find_best_model('VGG16')
    if model_path and os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        print(f"✅ Loaded VGG16 model: {os.path.basename(model_path)}")
        return model
    else:
        print("❌ VGG16 model not found!")
        return None

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

def convert_to_binary(y_true_multiclass, y_pred_multiclass):
    """
    Convert 4-class predictions to binary (Healthy vs Diseased)
    
    Mapping:
    - Class 0 (BrownSpot) → Diseased (1)
    - Class 1 (Healthy) → Healthy (0)
    - Class 2 (Hispa) → Diseased (1)
    - Class 3 (LeafBlast) → Diseased (1)
    """
    # Map class indices to binary
    # Healthy = 1 in original classes, maps to 0 in binary
    # All diseases (0, 2, 3) map to 1 in binary
    
    binary_map = {
        0: 1,  # BrownSpot → Diseased
        1: 0,  # Healthy → Healthy
        2: 1,  # Hispa → Diseased
        3: 1   # LeafBlast → Diseased
    }
    
    y_true_binary = np.array([binary_map[cls] for cls in y_true_multiclass])
    y_pred_binary = np.array([binary_map[cls] for cls in y_pred_multiclass])
    
    return y_true_binary, y_pred_binary

def evaluate_binary(model, val_generator):
    """Evaluate model in binary classification mode"""
    print("\n" + "="*70)
    print("BINARY CLASSIFICATION EVALUATION")
    print("="*70)
    print("Classes:")
    print("  0 = Healthy")
    print("  1 = Diseased (BrownSpot + Hispa + LeafBlast)")
    print("="*70)
    
    # Get predictions
    val_generator.reset()
    y_true_multi = val_generator.classes
    print(f"\n🔮 Making predictions on {len(y_true_multi)} validation images...")
    predictions = model.predict(val_generator, verbose=1)
    y_pred_multi = np.argmax(predictions, axis=1)
    
    # Convert to binary
    y_true_binary, y_pred_binary = convert_to_binary(y_true_multi, y_pred_multi)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    
    print("\n" + "="*70)
    print("BINARY CLASSIFICATION RESULTS")
    print("="*70)
    print(f"✅ Binary Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*70)
    
    # Classification report
    binary_class_names = ['Healthy', 'Diseased']
    report = classification_report(y_true_binary, y_pred_binary, 
                                   target_names=binary_class_names, 
                                   digits=4)
    print("\nClassification Report:")
    print("-" * 70)
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=binary_class_names, 
                yticklabels=binary_class_names,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 16})
    plt.title(f'Binary Classification - Confusion Matrix\nAccuracy: {accuracy:.2%}',
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    cm_path = os.path.join(config.RESULTS_DIR, 'binary_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Confusion matrix saved to: {cm_path}")
    plt.show()
    
    # Detailed breakdown
    print("\n" + "="*70)
    print("DETAILED BREAKDOWN")
    print("="*70)
    
    # Count samples
    healthy_true = np.sum(y_true_binary == 0)
    diseased_true = np.sum(y_true_binary == 1)
    
    healthy_pred_correct = cm[0, 0]
    diseased_pred_correct = cm[1, 1]
    
    print(f"\nHealthy Rice:")
    print(f"  Total samples: {healthy_true}")
    print(f"  Correctly identified: {healthy_pred_correct}")
    print(f"  Accuracy: {healthy_pred_correct/healthy_true:.2%}")
    
    print(f"\nDiseased Rice:")
    print(f"  Total samples: {diseased_true}")
    print(f"  Correctly identified: {diseased_pred_correct}")
    print(f"  Accuracy: {diseased_pred_correct/diseased_true:.2%}")
    
    # Save report
    report_path = os.path.join(config.RESULTS_DIR, 'binary_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("BINARY CLASSIFICATION EVALUATION\n")
        f.write("="*70 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write("Class Mapping:\n")
        f.write("  Healthy = Healthy class only\n")
        f.write("  Diseased = BrownSpot + Hispa + LeafBlast\n\n")
        f.write(report)
        f.write("\n\nDetailed Breakdown:\n")
        f.write("-"*70 + "\n")
        f.write(f"Healthy: {healthy_pred_correct}/{healthy_true} ({healthy_pred_correct/healthy_true:.2%})\n")
        f.write(f"Diseased: {diseased_pred_correct}/{diseased_true} ({diseased_pred_correct/diseased_true:.2%})\n")
    
    print(f"\n✅ Report saved to: {report_path}")
    
    return accuracy

def main():
    """Main binary evaluation function"""
    print("\n" + "="*70)
    print("RICE DISEASE DETECTION - BINARY CLASSIFICATION")
    print("="*70 + "\n")
    
    # Load model
    print("Loading VGG16 model...")
    model = load_model_for_binary()
    
    if model is None:
        print("\n❌ Cannot proceed without model!")
        return
    
    # Create validation generator
    print("\nLoading validation data...")
    val_generator = create_validation_generator()
    print(f"✅ Validation samples: {val_generator.samples}")
    
    # Evaluate
    accuracy = evaluate_binary(model, val_generator)
    
    print("\n" + "="*70)
    print("SUMMARY FOR MANAGER")
    print("="*70)
    print(f"\n📊 Binary Classification Accuracy: {accuracy*100:.1f}%")
    print("\nInterpretation:")
    print("  'Can the model distinguish healthy rice from diseased rice?'")
    print(f"  Answer: Yes, with {accuracy*100:.1f}% accuracy")
    print("\nThis is a more practical metric for real-world deployment")
    print("where the primary goal is disease detection.")
    print("="*70)
    
    print("\n✅ Binary classification evaluation complete!")
    print(f"📁 Results saved to: {config.RESULTS_DIR}")

if __name__ == "__main__":
    main()
