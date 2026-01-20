"""
Prepare existing VGG16 model for checkpoint resumption
This script converts your existing VGG16 model to the checkpoint format
"""

import os
import tensorflow as tf

# Configuration
MODELS_DIR = 'models'
EXISTING_MODEL = 'VGG16_20260113_135624_final.h5'  # 25 epochs completed
NEW_CHECKPOINT_NAME = 'VGG16_20260113_135624_checkpoint.keras'

def prepare_checkpoint():
    """Convert existing model to checkpoint format"""
    
    existing_path = os.path.join(MODELS_DIR, EXISTING_MODEL)
    checkpoint_path = os.path.join(MODELS_DIR, NEW_CHECKPOINT_NAME)
    
    print("\n" + "="*70)
    print("VGG16 CHECKPOINT PREPARATION")
    print("="*70 + "\n")
    
    # Check if existing model exists
    if not os.path.exists(existing_path):
        print(f"❌ Error: Model not found at {existing_path}")
        print("\nPlease check the model filename and try again.")
        return False
    
    print(f"✓ Found existing model: {EXISTING_MODEL}")
    print(f"  Model has 25 epochs completed")
    print(f"  Validation Accuracy: 50.6%\n")
    
    # Check if checkpoint already exists
    if os.path.exists(checkpoint_path):
        response = input(f"⚠ Checkpoint already exists. Overwrite? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Operation cancelled.")
            return False
    
    # Load and save the model in .keras format
    print(f"📋 Converting to checkpoint format: {NEW_CHECKPOINT_NAME}")
    
    try:
        # Load the .h5 model
        model = tf.keras.models.load_model(existing_path)
        # Save as .keras format
        model.save(checkpoint_path)
        print(f"✓ Model converted successfully!")
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        return False
    
    print(f"\n✓ Checkpoint created successfully!")
    print(f"✓ Original kept at: {EXISTING_MODEL}")
    print(f"✓ Checkpoint at: {NEW_CHECKPOINT_NAME}")
    
    print("\n" + "="*70)
    print("READY FOR CONTINUATION TRAINING!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run: python train.py --model vgg16 --epochs 50")
    print("  2. Training will resume from epoch 26")
    print("  3. You can stop anytime (Ctrl+C) and resume later!")
    print("\n")
    
    return True

if __name__ == "__main__":
    success = prepare_checkpoint()
    if success:
        print("✅ All done! You're ready to continue training.\n")
    else:
        print("❌ Preparation failed. Please check the errors above.\n")
