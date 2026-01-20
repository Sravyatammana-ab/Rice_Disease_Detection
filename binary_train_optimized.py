"""
OPTIMIZED Binary Classifier Training for Combined Dataset
Target: 85-92% accuracy with aggressive but smart fine-tuning
"""

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import config

def remap_to_binary(y):
    """Remap 4-class labels to 2-class (Healthy=0, Diseased=1)"""
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
        
        if class_weights is not None:
            class_indices = np.argmax(y_binary, axis=1)
            sample_weights = np.array([class_weights[idx] for idx in class_indices])
            yield x, y_binary, sample_weights
        else:
            yield x, y_binary

def calculate_class_weights(train_generator):
    """Calculate binary class weights"""
    class_counts = {}
    for class_name in config.CLASS_NAMES:
        class_idx = train_generator.class_indices[class_name]
        class_counts[class_name] = np.sum(train_generator.classes == class_idx)
    
    healthy_count = class_counts.get('Healthy', 0)
    diseased_count = sum([class_counts.get(c, 0) for c in ['BrownSpot', 'Hispa', 'LeafBlast']])
    
    total_samples = healthy_count + diseased_count
    weight_healthy = total_samples / (2 * healthy_count)
    weight_diseased = total_samples / (2 * diseased_count)
    
    print(f"\n📊 Binary Class Distribution:")
    print(f"  Healthy:  {healthy_count:4d} images → Weight: {weight_healthy:.4f}")
    print(f"  Diseased: {diseased_count:4d} images → Weight: {weight_diseased:.4f}")
    
    return {0: weight_healthy, 1: weight_diseased}

def build_optimized_binary_model():
    """Build ResNet50 binary classifier with optimal settings"""
    
    print("\n🏗️ Building OPTIMIZED ResNet50 Binary Classifier...")
    
    # Load ResNet50 base
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(*config.IMG_SIZE, 3),
        pooling='avg'
    )
    
    # UNFREEZE the top layers immediately for better learning
    # Freeze only the first 70% of layers (let last 30% adapt)
    total_layers = len(base_model.layers)
    freeze_until = int(total_layers * 0.7)
    
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True
    
    print(f"  ✅ Unfroze {total_layers - freeze_until}/{total_layers} layers from start")
    
    # Build classification head - simpler and more direct
    inputs = keras.Input(shape=(*config.IMG_SIZE, 3))
    x = base_model(inputs, training=True)  # training=True for better learning
    
    # Simpler head - less dropout for better gradient flow
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Binary output
    outputs = layers.Dense(2, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Higher learning rate for faster convergence
    optimizer = Adam(learning_rate=0.0005)  # 5x higher than before!
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    print(f"  ✅ Total parameters: {model.count_params():,}")
    print(f"  ✅ Trainable parameters: {sum([keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    
    return model, base_model

def train_optimized():
    """Train with optimized settings for high accuracy"""
    
    print("\n" + "="*70)
    print("🚀 OPTIMIZED BINARY TRAINING - TARGET: 85-92%")
    print("="*70)
    
    # Data augmentation - strong but not excessive
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Larger batch size for stability (32 instead of 16)
    train_gen = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=32,
        class_mode='categorical',
        classes=config.CLASS_NAMES,
        shuffle=True,
        seed=42
    )
    
    val_gen = val_datagen.flow_from_directory(
        config.VALIDATION_DIR,
        target_size=config.IMG_SIZE,
        batch_size=32,
        class_mode='categorical',
        classes=config.CLASS_NAMES,
        shuffle=False,
        seed=42
    )
    
    print(f"\n✅ Training samples: {train_gen.samples}")
    print(f"✅ Validation samples: {val_gen.samples}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_gen)
    
    # Build model
    model, base_model = build_optimized_binary_model()
    
    # Create binary generators
    train_binary = binary_data_generator(train_gen, class_weights)
    val_binary = binary_data_generator(val_gen)
    
    # Callbacks
    timestamp = '20260120_binary_optimized'
    
    checkpoint = ModelCheckpoint(
        os.path.join(config.MODELS_DIR, f'Binary_{timestamp}_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=15,  # Patient but not too much
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    csv_logger = CSVLogger(
        os.path.join(config.RESULTS_DIR, f'Binary_{timestamp}_log.csv')
    )
    
    callbacks = [checkpoint, early_stop, reduce_lr, csv_logger]
    
    # Single-stage training with unfrozen layers from start
    print("\n" + "="*70)
    print("🔥 TRAINING (Single Stage, Unfrozen from Start)")
    print("="*70)
    
    steps_per_epoch = len(train_gen)
    validation_steps = len(val_gen)
    
    history = model.fit(
        train_binary,
        steps_per_epoch=steps_per_epoch,
        epochs=50,  # Fewer epochs, faster convergence
        validation_data=val_binary,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    best_acc = max(history.history['val_accuracy'])
    print(f"\n🎯 Best validation accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    
    return model, history

def evaluate_model():
    """Evaluate the trained model"""
    
    # Find the model
    model_files = [f for f in os.listdir(config.MODELS_DIR) 
                   if f.startswith('Binary_20260120_binary_optimized') and f.endswith('.keras')]
    
    if not model_files:
        print("❌ No optimized binary model found!")
        return
    
    model_path = os.path.join(config.MODELS_DIR, model_files[0])
    print(f"\n✅ Loading: {model_files[0]}")
    
    model = keras.models.load_model(model_path)
    
    # Create validation generator
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_gen = val_datagen.flow_from_directory(
        config.VALIDATION_DIR,
        target_size=config.IMG_SIZE,
        batch_size=32,
        class_mode='categorical',
        classes=config.CLASS_NAMES,
        shuffle=False
    )
    
    val_binary = binary_data_generator(val_gen)
    
    print("\n🔮 Evaluating model...")
    results = model.evaluate(val_binary, steps=len(val_gen), verbose=1)
    
    print("\n" + "="*70)
    print("📊 OPTIMIZED BINARY CLASSIFIER RESULTS")
    print("="*70)
    print(f"✅ Accuracy:  {results[1]:.4f} ({results[1]*100:.2f}%)")
    print(f"✅ Precision: {results[2]:.4f}")
    print(f"✅ Recall:    {results[3]:.4f}")
    print("="*70)

def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--evaluate':
        evaluate_model()
    else:
        model, history = train_optimized()
        print("\n🎉 Training complete!")
        print("\nTo evaluate: python binary_train_optimized.py --evaluate")

if __name__ == "__main__":
    main()
