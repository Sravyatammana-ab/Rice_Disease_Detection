"""
Train models for rice disease detection
Supports multiple architectures: Custom CNN, VGG16, ResNet50, MobileNetV2, EfficientNetB0
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
)

import config
from data_loader import RiceDiseaseDataLoader
from models import RiceDiseaseModels, print_model_summary


def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics
    
    Args:
        history: Training history object
        save_path: Path to save the plot
    """
    # Get available metrics
    history_dict = history.history
    
    # Check if history has data
    if not history_dict or 'loss' not in history_dict or len(history_dict.get('loss', [])) == 0:
        print("\n⚠ Warning: No training history data available to plot.")
        print("  This can happen if training was stopped immediately or failed early.")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(history_dict['loss']) + 1)
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, history_dict['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[0, 1].plot(epochs, history_dict['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history_dict['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Precision (if available)
    if 'precision' in history_dict:
        axes[1, 0].plot(epochs, history_dict['precision'], 'b-', label='Training Precision', linewidth=2)
        axes[1, 0].plot(epochs, history_dict['val_precision'], 'r-', label='Validation Precision', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Precision', fontsize=12)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Recall (if available)
    if 'recall' in history_dict:
        axes[1, 1].plot(epochs, history_dict['recall'], 'b-', label='Training Recall', linewidth=2)
        axes[1, 1].plot(epochs, history_dict['val_recall'], 'r-', label='Validation Recall', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Recall', fontsize=12)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Training history plot saved to: {save_path}")
    
    # plt.show()  # Disabled - auto-save only
    return fig


def save_training_summary(history, model_name, save_dir):
    """
    Save training summary as JSON
    
    Args:
        history: Training history object
        model_name: Name of the model
        save_dir: Directory to save summary
    """
    # Check if history has data
    if not history.history or 'loss' not in history.history or len(history.history.get('loss', [])) == 0:
        print("\n⚠ Warning: No training history data available to save summary.")
        return None
    
    summary = {
        'model_name': model_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'epochs_trained': len(history.history['loss']),
        'final_metrics': {
            'train_loss': float(history.history['loss'][-1]),
            'train_accuracy': float(history.history['accuracy'][-1]),
            'val_loss': float(history.history['val_loss'][-1]),
            'val_accuracy': float(history.history['val_accuracy'][-1])
        },
        'best_metrics': {
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'best_val_loss': float(min(history.history['val_loss']))
        }
    }
    
    # Add precision and recall if available
    if 'precision' in history.history:
        summary['final_metrics']['train_precision'] = float(history.history['precision'][-1])
        summary['final_metrics']['val_precision'] = float(history.history['val_precision'][-1])
    
    if 'recall' in history.history:
        summary['final_metrics']['train_recall'] = float(history.history['recall'][-1])
        summary['final_metrics']['val_recall'] = float(history.history['val_recall'][-1])
    
    # Save to file
    summary_path = os.path.join(save_dir, f'{model_name}_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"✓ Training summary saved to: {summary_path}")
    
    return summary


def find_latest_checkpoint(model_type):
    """
    Find the latest checkpoint for a given model type
    
    Args:
        model_type: Type of model
        
    Returns:
        Tuple of (checkpoint_path, initial_epoch, model_filename)
    """
    # Search for existing checkpoints
    checkpoint_pattern = os.path.join(config.MODELS_DIR, f'{model_type}_*_checkpoint.keras')
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None, 0, None
    
    # Get the most recent checkpoint
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    
    # Extract model filename (without _checkpoint.keras)
    model_filename = os.path.basename(latest_checkpoint).replace('_checkpoint.keras', '')
    
    # Find corresponding CSV log to determine epoch count
    csv_log_path = os.path.join(config.RESULTS_DIR, f'{model_filename}_training_log.csv')
    
    initial_epoch = 0
    if os.path.exists(csv_log_path):
        # Read the CSV and get the last epoch number
        try:
            with open(csv_log_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]  # Remove empty lines
                if len(lines) > 1:  # Has header + at least one epoch
                    # Get the last data line (skip header)
                    last_line = lines[-1]
                    # First column is epoch number
                    epoch_num = last_line.split(',')[0]
                    initial_epoch = int(epoch_num) + 1  # Next epoch to train
        except Exception as e:
            print(f"\n⚠ Warning: Could not read epoch count from CSV: {e}")
            print("  Starting from epoch 0...")
            initial_epoch = 0
    
    return latest_checkpoint, initial_epoch, model_filename


def train_model(model_type='CustomCNN', epochs=None, batch_size=None, 
                learning_rate=None, use_augmentation=True):
    """
    Train a rice disease detection model with checkpoint resumption support
    
    Args:
        model_type: Type of model ('CustomCNN', 'VGG16', 'ResNet50', 'MobileNetV2', 'EfficientNetB0')
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        use_augmentation: Whether to use data augmentation
        
    Returns:
        Trained model and training history
    """
    print("\n" + "="*70)
    print("RICE DISEASE DETECTION - MODEL TRAINING")
    print("="*70 + "\n")
    
    # Use config defaults if not specified
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    learning_rate = learning_rate or config.LEARNING_RATE
    
    print(f"Training Configuration:")
    print(f"  Model Type: {model_type}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Data Augmentation: {use_augmentation}")
    print("-" * 70)
    
    # Step 1: Load Data
    print("\n[1/4] Loading Dataset...")
    data_loader = RiceDiseaseDataLoader(batch_size=batch_size)
    train_generator, val_generator = data_loader.get_generators(augmentation=use_augmentation)
    
    print(f"\n  Training samples: {train_generator.samples}")
    print(f"  Validation samples: {val_generator.samples}")
    print(f"  Steps per epoch: {train_generator.samples // batch_size}")
    print(f"  Validation steps: {val_generator.samples // batch_size}")
    
    # Step 2: Build/Load Model
    print(f"\n[2/4] Building Model: {model_type}...")
    
    # Map model names
    model_name_map = {
        'cnn': 'CustomCNN',
        'customcnn': 'CustomCNN',
        'vgg': 'VGG16',
        'vgg16': 'VGG16',
        'resnet': 'ResNet50',
        'resnet50': 'ResNet50',
        'mobilenet': 'MobileNetV2',
        'mobilenetv2': 'MobileNetV2',
        'efficientnet': 'EfficientNetB0',
        'efficientnetb0': 'EfficientNetB0'
    }
    
    model_type_normalized = model_name_map.get(model_type.lower(), model_type)
    
    # Check for existing checkpoint
    checkpoint_path, initial_epoch, model_filename = find_latest_checkpoint(model_type_normalized)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n  ✓ Found existing checkpoint: {os.path.basename(checkpoint_path)}")
        print(f"  ✓ Resuming from epoch {initial_epoch}")
        print(f"  ✓ Loading model...")
        
        # Load the checkpoint
        model = keras.models.load_model(checkpoint_path)
        print(f"  ✓ Model loaded successfully!")
        
    else:
        print(f"  ℹ No checkpoint found. Starting fresh training...")
        initial_epoch = 0
        
        # Create new model
        model_builder = RiceDiseaseModels()
        model = model_builder.get_model(
            model_name=model_type_normalized,
            compile_model=True,
            learning_rate=learning_rate
        )
        
        # Create new filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{model_type_normalized}_{timestamp}"
        
        print_model_summary(model)
    
    # Step 3: Setup Callbacks
    print(f"\n[3/4] Setting up Callbacks...")
    
    # Checkpoint - save after every epoch (for resumption)
    checkpoint_path = os.path.join(config.MODELS_DIR, f'{model_filename}_checkpoint.keras')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=False,  # Save every epoch
        mode='max',
        verbose=1
    )
    
    # Best Model Checkpoint - save only the best
    best_model_path = os.path.join(config.MODELS_DIR, f'{model_filename}_best.keras')
    best_checkpoint = ModelCheckpoint(
        best_model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # EarlyStopping - stop if no improvement
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        restore_best_weights=True,
        verbose=1
    )
    
    # ReduceLROnPlateau - reduce learning rate when stuck
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.REDUCE_LR_MIN_LR,
        verbose=1
    )
    
    # CSVLogger - log training metrics (append if resuming)
    csv_log_path = os.path.join(config.RESULTS_DIR, f'{model_filename}_training_log.csv')
    csv_logger = CSVLogger(csv_log_path, append=(initial_epoch > 0))
    
    callbacks = [checkpoint, best_checkpoint, early_stopping, reduce_lr, csv_logger]
    
    print(f"  ✓ Checkpoint (every epoch): {checkpoint_path}")
    print(f"  ✓ Best Model: {best_model_path}")
    print(f"  ✓ EarlyStopping: patience={config.EARLY_STOPPING_PATIENCE}")
    print(f"  ✓ ReduceLROnPlateau: patience={config.REDUCE_LR_PATIENCE}, factor={config.REDUCE_LR_FACTOR}")
    print(f"  ✓ CSVLogger: {csv_log_path} (append={initial_epoch > 0})")
    
    # Step 4: Train Model
    print(f"\n[4/4] Training Model...")
    if initial_epoch > 0:
        print(f"  ℹ Resuming training from epoch {initial_epoch + 1} to {epochs}")
    print("="*70)
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        initial_epoch=initial_epoch,  # Resume from this epoch
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    
    # Save final model
    final_model_path = os.path.join(config.MODELS_DIR, f'{model_filename}_final.keras')
    model.save(final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    # Plot and save training history
    plot_path = os.path.join(config.RESULTS_DIR, f'{model_filename}_training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    # Save training summary
    save_training_summary(history, model_filename, config.RESULTS_DIR)
    
    # Print final metrics (if available)
    if history.history and 'loss' in history.history and len(history.history.get('loss', [])) > 0:
        print("\nFinal Metrics:")
        print("-" * 70)
        print(f"  Training Loss: {history.history['loss'][-1]:.4f}")
        print(f"  Training Accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"  Validation Loss: {history.history['val_loss'][-1]:.4f}")
        print(f"  Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"  Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
        print("="*70 + "\n")
    else:
        print("\n⚠ Training completed but no new epoch metrics available.")
        print("  Check the CSV log for complete training history.")
        print("="*70 + "\n")
    
    return model, history


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description='Train Rice Disease Detection Model')
    parser.add_argument(
        '--model',
        type=str,
        default='CustomCNN',
        choices=['CustomCNN', 'VGG16', 'ResNet50', 'MobileNetV2', 'EfficientNetB0',
                 'cnn', 'vgg', 'resnet', 'mobilenet', 'efficientnet'],
        help='Model architecture to train'
    )
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Train model
    model, history = train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_augmentation=not args.no_augmentation
    )
    
    print("\n✓ Training pipeline completed successfully!")
    print(f"✓ Check the '{config.MODELS_DIR}' folder for saved models")
    print(f"✓ Check the '{config.RESULTS_DIR}' folder for training plots and logs\n")


if __name__ == "__main__":
    main()

