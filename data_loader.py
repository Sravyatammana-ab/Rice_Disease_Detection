"""
Data Loader Module for Rice Disease Detection
Handles data loading, preprocessing, augmentation, and generator creation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import config


class RiceDiseaseDataLoader:
    """
    Data loader class for rice disease detection
    Handles image loading, preprocessing, and data generation
    """
    
    def __init__(self, train_dir=None, validation_dir=None, 
                 img_size=None, batch_size=None):
        """
        Initialize data loader
        
        Args:
            train_dir: Path to training data directory
            validation_dir: Path to validation data directory
            img_size: Tuple of (height, width) for images
            batch_size: Batch size for data generators
        """
        self.train_dir = train_dir or config.TRAIN_DIR
        self.validation_dir = validation_dir or config.VALIDATION_DIR
        self.img_size = img_size or config.IMG_SIZE
        self.batch_size = batch_size or config.BATCH_SIZE
        
        self.class_names = config.CLASS_NAMES
        self.num_classes = config.NUM_CLASSES
        
        self.train_generator = None
        self.validation_generator = None
        
        print(f"Data Loader initialized:")
        print(f"  Train dir: {self.train_dir}")
        print(f"  Validation dir: {self.validation_dir}")
        print(f"  Image size: {self.img_size}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Classes: {self.class_names}")
    
    def create_train_generator(self, augmentation=True):
        """
        Create training data generator with optional augmentation
        
        Args:
            augmentation: Whether to apply data augmentation
            
        Returns:
            Training data generator
        """
        if augmentation:
            # Data augmentation for training
            train_datagen = ImageDataGenerator(
                rescale=config.RESCALE,
                rotation_range=config.AUGMENTATION_CONFIG['rotation_range'],
                width_shift_range=config.AUGMENTATION_CONFIG['width_shift_range'],
                height_shift_range=config.AUGMENTATION_CONFIG['height_shift_range'],
                shear_range=config.AUGMENTATION_CONFIG['shear_range'],
                zoom_range=config.AUGMENTATION_CONFIG['zoom_range'],
                horizontal_flip=config.AUGMENTATION_CONFIG['horizontal_flip'],
                vertical_flip=config.AUGMENTATION_CONFIG['vertical_flip'],
                fill_mode=config.AUGMENTATION_CONFIG['fill_mode']
            )
            print("Training generator created with data augmentation")
        else:
            # No augmentation, only rescaling
            train_datagen = ImageDataGenerator(rescale=config.RESCALE)
            print("Training generator created without augmentation")
        
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=True,
            seed=config.RANDOM_SEED
        )
        
        print(f"Found {self.train_generator.samples} training images")
        print(f"Class indices: {self.train_generator.class_indices}")
        
        return self.train_generator
    
    def create_validation_generator(self):
        """
        Create validation data generator (no augmentation)
        
        Returns:
            Validation data generator
        """
        # Only rescaling for validation, no augmentation
        validation_datagen = ImageDataGenerator(rescale=config.RESCALE)
        
        self.validation_generator = validation_datagen.flow_from_directory(
            self.validation_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=False,
            seed=config.RANDOM_SEED
        )
        
        print(f"Found {self.validation_generator.samples} validation images")
        print(f"Validation generator created")
        
        return self.validation_generator
    
    def get_generators(self, augmentation=True):
        """
        Create and return both train and validation generators
        
        Args:
            augmentation: Whether to apply augmentation to training data
            
        Returns:
            Tuple of (train_generator, validation_generator)
        """
        train_gen = self.create_train_generator(augmentation=augmentation)
        val_gen = self.create_validation_generator()
        
        return train_gen, val_gen
    
    def get_dataset_statistics(self):
        """
        Get statistics about the dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'train_samples': 0,
            'validation_samples': 0,
            'train_distribution': {},
            'validation_distribution': {}
        }
        
        # Count training images
        if os.path.exists(self.train_dir):
            for class_name in self.class_names:
                class_path = os.path.join(self.train_dir, class_name)
                if os.path.exists(class_path):
                    count = len([f for f in os.listdir(class_path) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    stats['train_distribution'][class_name] = count
                    stats['train_samples'] += count
        
        # Count validation images
        if os.path.exists(self.validation_dir):
            for class_name in self.class_names:
                class_path = os.path.join(self.validation_dir, class_name)
                if os.path.exists(class_path):
                    count = len([f for f in os.listdir(class_path) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    stats['validation_distribution'][class_name] = count
                    stats['validation_samples'] += count
        
        return stats
    
    def visualize_dataset_distribution(self, save_path=None):
        """
        Visualize the distribution of classes in train and validation sets
        
        Args:
            save_path: Path to save the plot (optional)
        """
        stats = self.get_dataset_statistics()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Training distribution
        if stats['train_distribution']:
            axes[0].bar(stats['train_distribution'].keys(), 
                       stats['train_distribution'].values(),
                       color='skyblue', edgecolor='navy')
            axes[0].set_title(f'Training Set Distribution\nTotal: {stats["train_samples"]} images', 
                            fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Class', fontsize=10)
            axes[0].set_ylabel('Number of Images', fontsize=10)
            axes[0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, (k, v) in enumerate(stats['train_distribution'].items()):
                axes[0].text(i, v, str(v), ha='center', va='bottom')
        
        # Validation distribution
        if stats['validation_distribution']:
            axes[1].bar(stats['validation_distribution'].keys(), 
                       stats['validation_distribution'].values(),
                       color='lightcoral', edgecolor='darkred')
            axes[1].set_title(f'Validation Set Distribution\nTotal: {stats["validation_samples"]} images', 
                            fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Class', fontsize=10)
            axes[1].set_ylabel('Number of Images', fontsize=10)
            axes[1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, (k, v) in enumerate(stats['validation_distribution'].items()):
                axes[1].text(i, v, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to: {save_path}")
        
        # plt.show()  # Disabled - auto-save only
        
        return stats
    
    def visualize_sample_images(self, num_images=12, save_path=None):
        """
        Display sample images from the training set
        
        Args:
            num_images: Number of images to display
            save_path: Path to save the plot (optional)
        """
        if self.train_generator is None:
            print("Creating temporary generator for visualization...")
            temp_gen = self.create_train_generator(augmentation=False)
        else:
            temp_gen = self.train_generator
        
        # Get a batch of images
        images, labels = next(temp_gen)
        num_images = min(num_images, len(images))
        
        # Calculate grid size
        cols = 4
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        axes = axes.flatten() if num_images > 1 else [axes]
        
        for i in range(num_images):
            axes[i].imshow(images[i])
            class_idx = np.argmax(labels[i])
            class_name = config.IDX_TO_CLASS[class_idx]
            axes[i].set_title(f'{class_name}', fontsize=10, fontweight='bold')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sample images saved to: {save_path}")
        
        # plt.show()  # Disabled - auto-save only
    
    def visualize_augmented_images(self, image_path=None, num_augmentations=9, 
                                   save_path=None):
        """
        Visualize augmented versions of a single image
        
        Args:
            image_path: Path to image file (if None, uses first training image)
            num_augmentations: Number of augmented versions to show
            save_path: Path to save the plot (optional)
        """
        # Load image
        if image_path is None:
            # Get first image from training set
            first_class = self.class_names[0]
            class_dir = os.path.join(self.train_dir, first_class)
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                print("No images found in training directory")
                return
            image_path = os.path.join(class_dir, images[0])
        
        # Load and preprocess image
        img = load_img(image_path, target_size=self.img_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Create augmentation generator
        datagen = ImageDataGenerator(
            rotation_range=config.AUGMENTATION_CONFIG['rotation_range'],
            width_shift_range=config.AUGMENTATION_CONFIG['width_shift_range'],
            height_shift_range=config.AUGMENTATION_CONFIG['height_shift_range'],
            shear_range=config.AUGMENTATION_CONFIG['shear_range'],
            zoom_range=config.AUGMENTATION_CONFIG['zoom_range'],
            horizontal_flip=config.AUGMENTATION_CONFIG['horizontal_flip'],
            vertical_flip=config.AUGMENTATION_CONFIG['vertical_flip'],
            fill_mode=config.AUGMENTATION_CONFIG['fill_mode']
        )
        
        # Generate augmented images
        cols = 3
        rows = (num_augmentations + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows))
        axes = axes.flatten() if num_augmentations > 1 else [axes]
        
        i = 0
        for batch in datagen.flow(img_array, batch_size=1):
            axes[i].imshow(batch[0].astype('uint8'))
            axes[i].set_title(f'Augmentation {i+1}', fontsize=10)
            axes[i].axis('off')
            i += 1
            if i >= num_augmentations:
                break
        
        # Hide unused subplots
        for i in range(num_augmentations, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Data Augmentation Examples', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Augmentation examples saved to: {save_path}")
        
        # plt.show()  # Disabled - auto-save only


def main():
    """Main function to test the data loader"""
    print("\n" + "="*70)
    print("RICE DISEASE DETECTION - DATA LOADER TEST")
    print("="*70 + "\n")
    
    # Initialize data loader
    data_loader = RiceDiseaseDataLoader()
    
    # Get dataset statistics
    print("\nDataset Statistics:")
    print("-" * 70)
    stats = data_loader.get_dataset_statistics()
    print(f"Training samples: {stats['train_samples']}")
    print(f"Training distribution: {stats['train_distribution']}")
    print(f"Validation samples: {stats['validation_samples']}")
    print(f"Validation distribution: {stats['validation_distribution']}")
    
    # Visualize dataset distribution
    if stats['train_samples'] > 0 or stats['validation_samples'] > 0:
        print("\nGenerating distribution plots...")
        data_loader.visualize_dataset_distribution(
            save_path=os.path.join(config.RESULTS_DIR, 'dataset_distribution.png')
        )
    
    # Create generators
    print("\nCreating data generators...")
    print("-" * 70)
    train_gen, val_gen = data_loader.get_generators(augmentation=True)
    
    # Visualize sample images
    if stats['train_samples'] > 0:
        print("\nVisualizing sample images...")
        data_loader.visualize_sample_images(
            num_images=12,
            save_path=os.path.join(config.RESULTS_DIR, 'sample_images.png')
        )
        
        # Visualize augmented images
        print("\nVisualizing data augmentation...")
        data_loader.visualize_augmented_images(
            num_augmentations=9,
            save_path=os.path.join(config.RESULTS_DIR, 'augmentation_examples.png')
        )
    else:
        print("\nNo images found. Please add images to the dataset directory:")
        print(f"  Training: {config.TRAIN_DIR}")
        print(f"  Validation: {config.VALIDATION_DIR}")
        print(f"  Expected subdirectories: {config.CLASS_NAMES}")
    
    print("\n" + "="*70)
    print("DATA LOADER TEST COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
