"""
Configuration file for Rice Disease Detection Project
Contains all hyperparameters, paths, and settings
"""

import os

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset selection - Use 'CombinedDataset' for maximum training data (5,447 images)
# or 'RiceDiseaseDataset' for original dataset (2,092 images)
DATASET_NAME = 'CombinedDataset'  # Changed from 'RiceDiseaseDataset' to 'CombinedDataset'

DATASET_DIR = os.path.join(BASE_DIR, 'dataset', DATASET_NAME)
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VALIDATION_DIR = os.path.join(DATASET_DIR, 'validation')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')

# Create directories if they don't exist
for directory in [MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
CLASS_NAMES = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']
NUM_CLASSES = len(CLASS_NAMES)

# Class to index mapping
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Normalization
RESCALE = 1.0 / 255.0

# ============================================================================
# DATA AUGMENTATION PARAMETERS (MODERATE FOR RESNET50)
# ============================================================================
AUGMENTATION_CONFIG = {
    'rotation_range': 20,  # Moderate rotation
    'width_shift_range': 0.15,  # Reduced from 0.25
    'height_shift_range': 0.15,  # Reduced from 0.25
    'shear_range': 0.15,  # Reduced from 0.25
    'zoom_range': 0.2,  # Reduced from 0.3
    'horizontal_flip': True,
    'vertical_flip': True,
    'brightness_range': [0.9, 1.1],  # Reduced from [0.8, 1.2]
    'fill_mode': 'nearest'
    # Removed channel_shift_range - too aggressive
}

# ============================================================================
# TRAINING HYPERPARAMETERS (BALANCED FOR SMALL DATASET)
# ============================================================================
BATCH_SIZE = 32  # Standard batch size for stability
EPOCHS = 50  # More epochs for convergence
LEARNING_RATE = 0.00001  # Very low LR for fine-tuning
VALIDATION_SPLIT = 0.2

# Early stopping (more patient)
EARLY_STOPPING_PATIENCE = 20  # Very patient for fine-tuning
EARLY_STOPPING_MIN_DELTA = 0.0001  # Small threshold

# Learning rate reduction
REDUCE_LR_PATIENCE = 10  # Very patient
REDUCE_LR_FACTOR = 0.5  # Gentle reduction
REDUCE_LR_MIN_LR = 1e-8

# Weight decay for regularization (reduced for small dataset)
WEIGHT_DECAY = 1e-6

# Label smoothing (reduced for small dataset)
LABEL_SMOOTHING = 0.05  # Mild smoothing

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Available models
AVAILABLE_MODELS = [
    'CustomCNN',
    'VGG16',
    'ResNet50',
    'MobileNetV2',
    'EfficientNetB0',
    'InceptionV3'
]

# Transfer learning configuration (FINE-TUNING ENABLED)
TRANSFER_LEARNING_CONFIG = {
    'include_top': False,
    'weights': 'imagenet',
    'pooling': 'avg',
    'fine_tune': True,  # ENABLED for fine-tuning
    'fine_tune_at': 10,  # Unfreeze last 10 layers for VGG16
    'dropout_rate': 0.3,  # Reduced from 0.5 for small dataset
    'l2_regularization': 0.01  # L2 regularization
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
# Model saving
MODEL_SAVE_FORMAT = 'h5'
CHECKPOINT_FILEPATH = os.path.join(MODELS_DIR, 'model_checkpoint_{epoch:02d}_{val_accuracy:.4f}.h5')
BEST_MODEL_FILEPATH = os.path.join(MODELS_DIR, 'best_model.h5')

# Results
TRAINING_HISTORY_PATH = os.path.join(RESULTS_DIR, 'training_history.png')
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
CLASSIFICATION_REPORT_PATH = os.path.join(RESULTS_DIR, 'classification_report.txt')
TRAINING_LOG_PATH = os.path.join(RESULTS_DIR, 'training_log.csv')
ROC_CURVE_PATH = os.path.join(RESULTS_DIR, 'roc_curves.png')

# ============================================================================
# RANDOM SEED
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================
VERBOSE = 1  # Training verbosity (0, 1, or 2)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def print_config():
    """Print current configuration"""
    print("=" * 70)
    print("RICE DISEASE DETECTION - CONFIGURATION")
    print("=" * 70)
    print(f"\nDataset Configuration:")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  Number of classes: {NUM_CLASSES}")
    print(f"  Train directory: {TRAIN_DIR}")
    print(f"  Validation directory: {VALIDATION_DIR}")
    
    print(f"\nImage Configuration:")
    print(f"  Image size: {IMG_SIZE}")
    print(f"  Input shape: {INPUT_SHAPE}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    
    print(f"\nOutput Directories:")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    print_config()
