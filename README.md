# Rice Disease Detection

A comprehensive deep learning project for detecting rice plant diseases using Convolutional Neural Networks (CNNs) and Transfer Learning with TensorFlow/Keras.

## 🌾 Overview

This project implements multiple state-of-the-art deep learning models to classify rice plant diseases into four categories:
- **BrownSpot**
- **Healthy**
- **Hispa**
- **LeafBlast**

## 📁 Project Structure

```
Rice_Disease_Detection/
│
├── dataset/
│   └── RiceDiseaseDataset/
│       ├── train/
│       │   ├── BrownSpot/
│       │   ├── Healthy/
│       │   ├── Hispa/
│       │   └── LeafBlast/
│       └── validation/
│           ├── BrownSpot/
│           ├── Healthy/
│           ├── Hispa/
│           └── LeafBlast/
│
├── models/              # Saved trained models
├── results/             # Training plots, metrics, logs
├── notebooks/           # Jupyter notebooks (optional)
│
├── config.py            # Configuration settings
├── data_loader.py       # Data loading and preprocessing
├── models.py            # Model architectures
├── train.py             # Training script
├── evaluate.py          # Evaluation script
├── ensemble.py          # Ensemble prediction
└── requirements.txt     # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your rice disease images in the following structure:
```
dataset/RiceDiseaseDataset/
├── train/
│   ├── BrownSpot/
│   ├── Healthy/
│   ├── Hispa/
│   └── LeafBlast/
└── validation/
    ├── BrownSpot/
    ├── Healthy/
    ├── Hispa/
    └── LeafBlast/
```

### 3. Test Data Loading

```bash
python data_loader.py
```

### 4. Train a Model

```bash
# Train Custom CNN (default)
python train.py

# Train VGG16
python train.py --model VGG16

# Train ResNet50
python train.py --model ResNet50

# Train MobileNetV2
python train.py --model MobileNetV2

# Train EfficientNetB0
python train.py --model EfficientNetB0
```

### 5. Evaluate Model

```bash
python evaluate.py --model models/VGG16_best.h5
```

### 6. Ensemble Prediction

```bash
# Soft voting
python ensemble.py --models models/model1.h5 models/model2.h5 --method soft

# Hard voting
python ensemble.py --models models/model1.h5 models/model2.h5 --method hard

# Compare all methods
python ensemble.py --models models/*.h5 --method compare
```

## 🎯 Features

### Data Processing
- ✅ Automatic data loading from directory structure
- ✅ Image preprocessing and normalization
- ✅ Data augmentation (rotation, shift, flip, zoom)
- ✅ Train/validation split handling
- ✅ Visualization utilities

### Model Architectures
- ✅ **Custom CNN** - 4 convolutional blocks with batch normalization
- ✅ **VGG16** - Transfer learning with ImageNet weights
- ✅ **ResNet50** - Deep residual network
- ✅ **MobileNetV2** - Lightweight mobile architecture
- ✅ **EfficientNetB0** - Efficient scaling architecture

### Training Features
- ✅ **ModelCheckpoint** - Saves best model based on validation accuracy
- ✅ **EarlyStopping** - Stops training when no improvement
- ✅ **ReduceLROnPlateau** - Reduces learning rate when stuck
- ✅ **CSVLogger** - Logs all metrics to CSV
- ✅ Training visualization (loss, accuracy, precision, recall)
- ✅ Training summary export (JSON)

### Evaluation Metrics
- ✅ Confusion matrix
- ✅ Classification report (precision, recall, f1-score)
- ✅ ROC curves and AUC scores
- ✅ Per-class accuracy
- ✅ Sample predictions visualization
- ✅ Model comparison

### Ensemble Methods
- ✅ Soft voting (average probabilities)
- ✅ Hard voting (majority vote)
- ✅ Weighted averaging
- ✅ Method comparison

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Dataset paths
TRAIN_DIR = 'dataset/RiceDiseaseDataset/train'
VALIDATION_DIR = 'dataset/RiceDiseaseDataset/validation'

# Classes
CLASS_NAMES = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']
NUM_CLASSES = 4

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Training
EPOCHS = 20
LEARNING_RATE = 0.0001
```

## 📊 Training Options

### Command-Line Arguments

**train.py:**
```bash
--model           # Model type (CustomCNN, VGG16, ResNet50, MobileNetV2, EfficientNetB0)
--epochs          # Number of training epochs (default: 20)
--batch-size      # Batch size (default: 32)
--lr              # Learning rate (default: 0.0001)
--no-augmentation # Disable data augmentation
```

**evaluate.py:**
```bash
--model           # Path to saved model (.h5)
--compare         # Paths to multiple models for comparison
```

**ensemble.py:**
```bash
--models          # Paths to saved models (.h5)
--method          # Ensemble method (soft, hard, weighted, compare)
--weights         # Weights for weighted average
```

## 📈 Example Usage

### Train Multiple Models

```bash
# Train Custom CNN
python train.py --model CustomCNN --epochs 20

# Train VGG16
python train.py --model VGG16 --epochs 20

# Train ResNet50
python train.py --model ResNet50 --epochs 20

# Train MobileNetV2 with custom settings
python train.py --model MobileNetV2 --epochs 30 --batch-size 16 --lr 0.00001
```

### Evaluate and Compare

```bash
# Evaluate single model
python evaluate.py --model models/VGG16_best.h5

# Compare multiple models
python evaluate.py --compare models/CustomCNN_best.h5 models/VGG16_best.h5 models/ResNet50_best.h5
```

### Ensemble Prediction

```bash
# Compare all ensemble methods
python ensemble.py --models models/VGG16_best.h5 models/ResNet50_best.h5 models/MobileNetV2_best.h5 --method compare

# Use weighted ensemble
python ensemble.py --models model1.h5 model2.h5 model3.h5 --method weighted --weights 0.5 0.3 0.2
```

## 📁 Output Files

After training and evaluation, you'll find:

**models/**
- `ModelName_TIMESTAMP_best.h5` - Best model checkpoint
- `ModelName_TIMESTAMP_final.h5` - Final model

**results/**
- `ModelName_training_history.png` - Training plots
- `ModelName_training_log.csv` - Training metrics
- `ModelName_training_summary.json` - Summary statistics
- `confusion_matrix.png` - Confusion matrix
- `roc_curves.png` - ROC curves
- `classification_report.txt` - Detailed report
- `prediction_samples.png` - Sample predictions
- `ensemble_METHOD_results.json` - Ensemble results

## 🔧 Troubleshooting

### Dataset Not Found
Ensure your dataset is in the correct structure:
```
dataset/RiceDiseaseDataset/train/BrownSpot/
dataset/RiceDiseaseDataset/train/Healthy/
dataset/RiceDiseaseDataset/train/Hispa/
dataset/RiceDiseaseDataset/train/LeafBlast/
```

### Out of Memory
- Reduce `BATCH_SIZE` in `config.py`
- Use `--batch-size 16` or smaller
- Use MobileNetV2 instead of heavier models

### Low Accuracy
- Increase `EPOCHS` (e.g., 50-100)
- Try different models
- Ensure dataset is balanced
- Use ensemble methods

## 🧪 Testing Individual Modules

```bash
# Test configuration
python config.py

# Test data loader
python data_loader.py

# Test model building
python models.py
```

## 📚 Dependencies

- Python 3.8+
- TensorFlow 2.12.0
- Keras 2.12.0
- NumPy 1.23.5
- Pandas 1.5.3
- Matplotlib 3.7.1
- Seaborn 0.12.2
- scikit-learn 1.2.2
- OpenCV 4.7.0

## 🎓 Model Performance Tips

1. **Start with Custom CNN** - Fast training, baseline performance
2. **Try VGG16** - Good balance of speed and accuracy
3. **Use ResNet50** - Better for complex patterns
4. **MobileNetV2** - Best for resource-constrained environments
5. **Ensemble** - Combine multiple models for best results

## 📝 Citation

If you use this project in your research, please cite:

```
Rice Disease Detection using Deep Learning
Authors: [Your Name]
Year: 2026
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Happy Training! 🌾🚀**
#   R i c e _ D i s e a s e _ D e t e c t i o n  
 #   R i c e _ D i s e a s e _ D e t e c t i o n  
 