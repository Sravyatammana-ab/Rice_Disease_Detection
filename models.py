"""
Model Architectures for Rice Disease Detection
Includes custom CNN and transfer learning models
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNetV2, EfficientNetB0, InceptionV3
)
from tensorflow.keras.optimizers import Adam
import config


class RiceDiseaseModels:
    """
    Model builder class for rice disease detection
    Supports custom CNN and various transfer learning architectures
    """
    
    def __init__(self, input_shape=None, num_classes=None):
        """
        Initialize model builder
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape or config.INPUT_SHAPE
        self.num_classes = num_classes or config.NUM_CLASSES
        
        print(f"Model Builder initialized:")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Number of classes: {self.num_classes}")
    
    def build_custom_cnn(self, name="CustomCNN"):
        """
        Build a custom CNN architecture for rice disease classification
        
        Args:
            name: Model name
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential(name=name)
        
        # First Convolutional Block
        model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                               input_shape=self.input_shape, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Second Convolutional Block
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Third Convolutional Block
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Fourth Convolutional Block
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Dense Layers
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        
        # Output Layer
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        print(f"\n{'='*70}")
        print(f"Custom CNN Model Created: {name}")
        print(f"{'='*70}")
        
        return model
    
    def build_transfer_learning_model(self, base_model_name='VGG16', 
                                      fine_tune=False, 
                                      fine_tune_at=100):
        """
        Build OPTIMIZED transfer learning model for >90% accuracy
        
        Args:
            base_model_name: Name of base model ('VGG16', 'ResNet50', etc.)
            fine_tune: Whether to fine-tune the base model
            fine_tune_at: Layer index to start fine-tuning from
            
        Returns:
            Compiled Keras model
        """
        # Select base model
        base_models = {
            'VGG16': VGG16,
            'ResNet50': ResNet50,
            'MobileNetV2': MobileNetV2,
            'EfficientNetB0': EfficientNetB0,
            'InceptionV3': InceptionV3
        }
        
        if base_model_name not in base_models:
            raise ValueError(f"Unknown model: {base_model_name}. "
                           f"Available: {list(base_models.keys())}")
        
        # Create base model
        base_model = base_models[base_model_name](
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling='avg'
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build complete model with optimized architecture
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        
        # Base model
        x = base_model(x, training=False)
        
        # OPTIMIZED top layers for >90% accuracy
        # Global Average Pooling already in base model (pooling='avg')
        
        # Dense Block 1 with balanced regularization
        x = layers.Dense(1024, kernel_regularizer=keras.regularizers.l2(config.WEIGHT_DECAY))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)  # Reduced from 0.5
        
        # Dense Block 2
        x = layers.Dense(512, kernel_regularizer=keras.regularizers.l2(config.WEIGHT_DECAY))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)  # Reduced from 0.4
        
        # Dense Block 3  
        x = layers.Dense(256, kernel_regularizer=keras.regularizers.l2(config.WEIGHT_DECAY))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.1)(x)  # Reduced from 0.3
        
        # Output layer with label smoothing (applied in loss)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs, name=f"{base_model_name}_Optimized")
        
        # Fine-tuning with progressive unfreezing
        if fine_tune:
            base_model.trainable = True
            # Freeze only early layers
            total_layers = len(base_model.layers)
            freeze_until = max(0, total_layers - fine_tune_at)
            
            for i, layer in enumerate(base_model.layers):
                if i < freeze_until:
                    layer.trainable = False
                else:
                    layer.trainable = True
            
            trainable_layers = sum([layer.trainable for layer in base_model.layers])
            print(f"Fine-tuning: {trainable_layers}/{total_layers} layers trainable")
        
        print(f"\n{'='*70}")
        print(f"OPTIMIZED Transfer Learning Model: {base_model_name}")
        print(f"Base model trainable: {base_model.trainable}")
        if fine_tune:
            trainable_layers = sum([layer.trainable for layer in base_model.layers])
            print(f"Trainable base layers: {trainable_layers}/{len(base_model.layers)}")
        print(f"Target: >90% accuracy")
        print(f"{'='*70}")
        
        return model
    
    def compile_model(self, model, learning_rate=None, metrics=None):
        """
        Compile model with OPTIMIZED settings for >90% accuracy
        Uses AdamW optimizer with weight decay and label smoothing
        
        Args:
            model: Keras model to compile
            learning_rate: Learning rate for optimizer
            metrics: List of metrics to track
            
        Returns:
            Compiled model
        """
        lr = learning_rate or config.LEARNING_RATE
        
        if metrics is None:
            metrics = [
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
            ]
        
        # Use AdamW (Adam with weight decay) for better generalization
        from tensorflow.keras.optimizers import AdamW
        optimizer = AdamW(
            learning_rate=lr,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Use label smoothing to prevent overconfidence
        loss = keras.losses.CategoricalCrossentropy(
            label_smoothing=config.LABEL_SMOOTHING
        )
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        print(f"\nOPTIMIZED Model Compilation:")
        print(f"  Optimizer: AdamW (lr={lr}, weight_decay={config.WEIGHT_DECAY})")
        print(f"  Loss: Categorical Crossentropy (label_smoothing={config.LABEL_SMOOTHING})")
        print(f"  Metrics: {[m if isinstance(m, str) else m.name for m in metrics]}")
        print(f"  Target: >90% validation accuracy")
        
        return model
    
    def get_model(self, model_name='CustomCNN', compile_model=True, 
                  fine_tune=False, learning_rate=None):
        """
        Get a model by name (convenience method)
        
        Args:
            model_name: Name of model to build
            compile_model: Whether to compile the model
            fine_tune: Whether to fine-tune (for transfer learning)
            learning_rate: Learning rate for optimizer
            
        Returns:
            Keras model (compiled if compile_model=True)
        """
        print(f"\nBuilding model: {model_name}")
        print("-" * 70)
        
        if model_name == 'CustomCNN':
            model = self.build_custom_cnn()
        elif model_name in config.AVAILABLE_MODELS[1:]:  # Transfer learning models
            model = self.build_transfer_learning_model(
                base_model_name=model_name,
                fine_tune=fine_tune,
                fine_tune_at=config.TRANSFER_LEARNING_CONFIG['fine_tune_at']
            )
        else:
            raise ValueError(f"Unknown model: {model_name}. "
                           f"Available: {config.AVAILABLE_MODELS}")
        
        if compile_model:
            model = self.compile_model(model, learning_rate=learning_rate)
        
        return model


def print_model_summary(model, save_path=None):
    """
    Print and optionally save model summary
    
    Args:
        model: Keras model
        save_path: Path to save summary (optional)
    """
    print(f"\n{'='*70}")
    print(f"MODEL SUMMARY: {model.name}")
    print(f"{'='*70}")
    model.summary()
    
    # Count parameters
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    print(f"\n{'='*70}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"{'='*70}\n")
    
    if save_path:
        with open(save_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write(f"\nTotal parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n")
            f.write(f"Non-trainable parameters: {non_trainable_params:,}\n")
        print(f"Model summary saved to: {save_path}")


def visualize_model_architecture(model, save_path=None):
    """
    Visualize model architecture
    
    Args:
        model: Keras model
        save_path: Path to save visualization
    """
    try:
        keras.utils.plot_model(
            model,
            to_file=save_path or 'model_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=96
        )
        print(f"Model architecture saved to: {save_path}")
    except Exception as e:
        print(f"Could not visualize model: {e}")
        print("Install pydot and graphviz for model visualization")


def main():
    """Main function to test model building"""
    import os
    
    print("\n" + "="*70)
    print("RICE DISEASE DETECTION - MODEL BUILDER TEST")
    print("="*70 + "\n")
    
    # Initialize model builder
    model_builder = RiceDiseaseModels()
    
    # Test 1: Custom CNN
    print("\n" + "="*70)
    print("TEST 1: Building Custom CNN")
    print("="*70)
    cnn_model = model_builder.get_model('CustomCNN', compile_model=True)
    print_model_summary(cnn_model, 
                       save_path=os.path.join(config.RESULTS_DIR, 'custom_cnn_summary.txt'))
    
    # Test 2: Transfer Learning Models
    transfer_models = ['VGG16', 'ResNet50', 'MobileNetV2']
    
    for model_name in transfer_models:
        print("\n" + "="*70)
        print(f"TEST: Building {model_name}")
        print("="*70)
        try:
            tl_model = model_builder.get_model(model_name, compile_model=True, fine_tune=False)
            print_model_summary(tl_model,
                              save_path=os.path.join(config.RESULTS_DIR, 
                                                     f'{model_name.lower()}_summary.txt'))
            print(f"✓ {model_name} built successfully")
        except Exception as e:
            print(f"✗ Error building {model_name}: {e}")
    
    print("\n" + "="*70)
    print("MODEL BUILDER TEST COMPLETED")
    print(f"Available models: {config.AVAILABLE_MODELS}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
