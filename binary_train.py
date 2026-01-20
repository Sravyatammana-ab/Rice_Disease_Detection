"""
Dedicated Binary Classifier Training
Train a model specifically for Healthy vs Diseased classification
"""

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import config

def create_binary_data_generators():
    """Create data generators with binary labels"""
    
    # Heavy augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create binary directory structure mapping
    # BrownSpot (0), Hispa (2), LeafBlast (3) → Diseased
    # Healthy (1) → Healthy
    
    print("\n📊 Creating binary data generators...")
    print("Mapping:")
    print("  BrownSpot → Diseased")
    print("  Healthy → Healthy")
    print("  Hispa → Diseased")
    print("  LeafBlast → Diseased")
    
    # We'll use a custom generator that remaps on-the-fly
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=32,
        class_mode='categorical',
        classes=config.CLASS_NAMES,
        shuffle=True,
        seed=42
    )
    
    val_generator = val_datagen.flow_from_directory(
        config.VALIDATION_DIR,
        target_size=config.IMG_SIZE,
        batch_size=32,
        class_mode='categorical',
        classes=config.CLASS_NAMES,
        shuffle=False,
        seed=42
    )
    
    return train_generator, val_generator

def remap_to_binary(y):
    """Remap 4-class labels to 2-class (Healthy=0, Diseased=1)"""
    # Original: [BrownSpot, Healthy, Hispa, LeafBlast]
    # New: [Healthy, Diseased]
    
    binary_labels = np.zeros((y.shape[0], 2))
    class_indices = np.argmax(y, axis=1)
    
    for i, cls in enumerate(class_indices):
        if cls == 1:  # Healthy
            binary_labels[i, 0] = 1
        else:  # All diseases
            binary_labels[i, 1] = 1
    
    return binary_labels

def binary_data_generator(generator, class_weights=None):
    """Wrapper to convert multi-class generator to binary with sample weights"""
    while True:
        x, y = next(generator)
        y_binary = remap_to_binary(y)
        
        # If class weights provided, create sample weights
        if class_weights is not None:
            # Get the class index for each sample (0=Healthy, 1=Diseased)
            class_indices = np.argmax(y_binary, axis=1)
            # Map to sample weights
            sample_weights = np.array([class_weights[idx] for idx in class_indices])
            yield x, y_binary, sample_weights
        else:
            yield x, y_binary


def build_binary_classifier(base_model='ResNet50'):
    """Build a dedicated binary classifier"""
    
    if base_model == 'ResNet50':
        from tensorflow.keras.applications import ResNet50
        base = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(*config.IMG_SIZE, 3),
            pooling='avg'
        )
    elif base_model == 'MobileNetV2':
        base = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(*config.IMG_SIZE, 3),
            pooling='avg'
        )
    else:  # VGG16
        base = VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=(*config.IMG_SIZE, 3),
            pooling='avg'
        )
    
    # Freeze base initially
    base.trainable = False
    
    # Build binary classifier head
    inputs = keras.Input(shape=(*config.IMG_SIZE, 3))
    x = base(inputs, training=False)
    
    # Deeper head for binary classification (better for ResNet50)
    x = layers.Dense(512, kernel_regularizer=keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256, kernel_regularizer=keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, kernel_regularizer=keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Binary output
    outputs = layers.Dense(2, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile
    optimizer = AdamW(learning_rate=0.0001, weight_decay=1e-5)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    return model, base

def calculate_binary_class_weights(train_generator):
    """Calculate class weights for binary classification to handle imbalance"""
    # Count samples per class in the original 4-class dataset
    class_counts = {}
    for class_name in config.CLASS_NAMES:
        class_idx = train_generator.class_indices[class_name]
        class_counts[class_name] = np.sum(train_generator.classes == class_idx)
    
    # Calculate binary class distribution
    healthy_count = class_counts.get('Healthy', 0)
    diseased_count = sum([class_counts.get(c, 0) for c in ['BrownSpot', 'Hispa', 'LeafBlast']])
    
    total_samples = healthy_count + diseased_count
    
    # Calculate weights using sklearn's balanced approach
    # weight = n_samples / (n_classes * n_samples_for_class)
    weight_healthy = total_samples / (2 * healthy_count)
    weight_diseased = total_samples / (2 * diseased_count)
    
    print(f"\n📊 Class Distribution for Binary Classification:")
    print(f"  Healthy:  {healthy_count:4d} images → Weight: {weight_healthy:.4f}")
    print(f"  Diseased: {diseased_count:4d} images → Weight: {weight_diseased:.4f}")
    
    # Return as dictionary for Keras (indices: 0=Healthy, 1=Diseased)
    class_weights = {
        0: weight_healthy,
        1: weight_diseased
    }
    
    return class_weights


def train_binary_classifier(epochs=100):
    """Train dedicated binary classifier with AGGRESSIVE settings for 90% target"""
    
    print("\n" + "="*70)
    print("AGGRESSIVE BINARY CLASSIFIER TRAINING - TARGET: 90%")
    print("="*70)
    
    # Create generators with HEAVY augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,  # Increased
        width_shift_range=0.3,  # Increased
        height_shift_range=0.3,  # Increased
        shear_range=0.3,  # Increased
        zoom_range=0.35,  # Increased
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],  # More variation
        channel_shift_range=25.0,  # Color variation
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=16,  # Smaller for better gradients
        class_mode='categorical',
        classes=config.CLASS_NAMES,
        shuffle=True,
        seed=42
    )
    
    val_gen = val_datagen.flow_from_directory(
        config.VALIDATION_DIR,
        target_size=config.IMG_SIZE,
        batch_size=16,
        class_mode='categorical',
        classes=config.CLASS_NAMES,
        shuffle=False,
        seed=42
    )
    
    print(f"\n✅ Training samples: {train_gen.samples}")
    print(f"✅ Validation samples: {val_gen.samples}")
    
    # Calculate class weights for handling imbalance
    class_weights = calculate_binary_class_weights(train_gen)
    
    # Build model - using ResNet50 (best accuracy)
    print("\n🏗️ Building ResNet50 binary classifier...")
    model, base_model = build_binary_classifier('ResNet50')
    
    print(f"✅ Model built - Parameters: {model.count_params():,}")
    
    # Create binary generators WITH CLASS WEIGHTS
    train_binary = binary_data_generator(train_gen, class_weights)
    val_binary = binary_data_generator(val_gen)  # No weights for validation
    
    # STAGE 1: Train classifier head only (40 epochs)
    print("\n" + "="*70)
    print("STAGE 1: Training classifier head (40 epochs) WITH CLASS WEIGHTS")
    print("="*70)
    
    timestamp = '20260120_binary_combined'
    
    checkpoint = ModelCheckpoint(
        os.path.join(config.MODELS_DIR, f'Binary_{timestamp}_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,  # Aggressive reduction
        patience=5,
        min_lr=1e-8,
        verbose=1
    )
    
    csv_logger = CSVLogger(
        os.path.join(config.RESULTS_DIR, f'Binary_{timestamp}_log.csv')
    )
    
    callbacks_stage1 = [checkpoint, reduce_lr, csv_logger]
    
    steps_per_epoch = len(train_gen)
    validation_steps = len(val_gen)
    
    # STAGE 1 training (class weights in generator)
    history1 = model.fit(
        train_binary,
        steps_per_epoch=steps_per_epoch,
        epochs=40,
        validation_data=val_binary,
        validation_steps=validation_steps,
        callbacks=callbacks_stage1,
        verbose=1
    )
    
    print(f"\n✅ Stage 1 complete. Best val_acc: {max(history1.history['val_accuracy']):.4f}")
    
    # STAGE 2: Fine-tune with unfrozen layers (60 more epochs)
    print("\n" + "="*70)
    print("STAGE 2: Fine-tuning with unfrozen layers (60 epochs)")
    print("="*70)
    
    # Unfreeze top layers
    base_model.trainable = True
    
    # Freeze first 80% of layers, unfreeze last 20%
    total_layers = len(base_model.layers)
    freeze_until = int(total_layers * 0.8)
    
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True
    
    print(f"Unfroze {total_layers - freeze_until} layers out of {total_layers}")
    
    # Recompile with lower learning rate for fine-tuning
    optimizer = AdamW(learning_rate=0.00001, weight_decay=1e-6)  # 10x lower
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=25,  # Very patient for fine-tuning
        restore_best_weights=True,
        verbose=1
    )
    
    callbacks_stage2 = [checkpoint, early_stop, reduce_lr, csv_logger]
    
    # STAGE 2 training (class weights in generator)
    history2 = model.fit(
        train_binary,
        steps_per_epoch=steps_per_epoch,
        epochs=60,
        initial_epoch=40,
        validation_data=val_binary,
        validation_steps=validation_steps,
        callbacks=callbacks_stage2,
        verbose=1
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    best_acc = max(all_val_acc)
    
    print(f"✅ Best validation accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print("="*70)
    
    return model, [history1, history2]

def evaluate_binary_model():
    """Evaluate the trained binary model"""
    
    # Find binary model
    model_files = [f for f in os.listdir(config.MODELS_DIR) if f.startswith('Binary_') and f.endswith('_best.keras')]
    
    if not model_files:
        print("❌ No binary model found! Train first.")
        return
    
    model_files.sort(reverse=True)
    model_path = os.path.join(config.MODELS_DIR, model_files[0])
    
    print(f"\n✅ Loading: {model_files[0]}")
    model = keras.models.load_model(model_path)
    
    # Create validation generator
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        config.VALIDATION_DIR,
        target_size=config.IMG_SIZE,
        batch_size=32,
        class_mode='categorical',
        classes=config.CLASS_NAMES,
        shuffle=False,
        seed=42
    )
    
    # Create binary generator
    val_binary = binary_data_generator(val_generator)
    
    # Evaluate
    print("\n🔮 Evaluating model...")
    val_generator.reset()
    results = model.evaluate(val_binary, steps=len(val_generator), verbose=1)
    
    print("\n" + "="*70)
    print("DEDICATED BINARY CLASSIFIER RESULTS")
    print("="*70)
    print(f"✅ Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
    print(f"✅ Precision: {results[2]:.4f}")
    print(f"✅ Recall: {results[3]:.4f}")
    print(f"✅ AUC: {results[4]:.4f}")
    print("="*70)

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--evaluate':
        evaluate_binary_model()
    else:
        model, history = train_binary_classifier(epochs=60)
        print("\n\n🎉 Training complete!")
        print("\nTo evaluate: python binary_train.py --evaluate")

if __name__ == "__main__":
    main()
