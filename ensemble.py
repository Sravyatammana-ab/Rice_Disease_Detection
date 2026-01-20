"""
Ensemble methods for combining multiple models
Supports voting, weighted averaging, and stacking
"""

import os
import argparse
import numpy as np
import json
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

import config
from data_loader import RiceDiseaseDataLoader
from sklearn.metrics import accuracy_score, classification_report


class EnsemblePredictor:
    """
    Ensemble predictor class for combining multiple models
    """
    
    def __init__(self, model_paths):
        """
        Initialize ensemble predictor
        
        Args:
            model_paths: List of paths to saved models
        """
        self.model_paths = model_paths
        self.models = []
        self.load_models()
    
    def load_models(self):
        """Load all models"""
        print(f"\nLoading {len(self.model_paths)} models...")
        print("-" * 70)
        
        for i, model_path in enumerate(self.model_paths):
            try:
                model = keras.models.load_model(model_path)
                self.models.append(model)
                model_name = os.path.basename(model_path)
                print(f"  [{i+1}] ✓ {model_name}")
            except Exception as e:
                print(f"  [{i+1}] ✗ Failed to load {model_path}: {e}")
        
        print(f"\n✓ Successfully loaded {len(self.models)} models")
    
    def predict_soft_voting(self, data):
        """
        Soft voting: Average predicted probabilities
        
        Args:
            data: Input data (generator or numpy array)
            
        Returns:
            Averaged predictions
        """
        print("\nUsing Soft Voting (Average Probabilities)...")
        
        all_predictions = []
        
        for i, model in enumerate(self.models):
            print(f"  Getting predictions from model {i+1}...")
            predictions = model.predict(data, verbose=0)
            all_predictions.append(predictions)
        
        # Average probabilities
        avg_predictions = np.mean(all_predictions, axis=0)
        
        return avg_predictions
    
    def predict_hard_voting(self, data):
        """
        Hard voting: Majority vote on class predictions
        
        Args:
            data: Input data (generator or numpy array)
            
        Returns:
            Voted predictions
        """
        print("\nUsing Hard Voting (Majority Vote)...")
        
        all_predictions = []
        
        for i, model in enumerate(self.models):
            print(f"  Getting predictions from model {i+1}...")
            predictions = model.predict(data, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            all_predictions.append(predicted_classes)
        
        # Stack predictions and get majority vote
        all_predictions = np.array(all_predictions)  # Shape: (n_models, n_samples)
        
        # Majority vote
        voted_predictions = []
        for i in range(all_predictions.shape[1]):
            votes = all_predictions[:, i]
            # Get most common vote
            unique, counts = np.unique(votes, return_counts=True)
            majority_vote = unique[np.argmax(counts)]
            voted_predictions.append(majority_vote)
        
        voted_predictions = np.array(voted_predictions)
        
        # Convert to one-hot probabilities (1.0 for voted class, 0.0 for others)
        n_classes = config.NUM_CLASSES
        probabilities = np.zeros((len(voted_predictions), n_classes))
        probabilities[np.arange(len(voted_predictions)), voted_predictions] = 1.0
        
        return probabilities
    
    def predict_weighted_average(self, data, weights=None):
        """
        Weighted average of predictions
        
        Args:
            data: Input data (generator or numpy array)
            weights: List of weights for each model (default: equal weights)
            
        Returns:
            Weighted averaged predictions
        """
        print("\nUsing Weighted Average...")
        
        if weights is None:
            weights = [1.0 / len(self.models)] * len(self.models)
        else:
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
        
        print(f"  Weights: {weights}")
        
        all_predictions = []
        
        for i, model in enumerate(self.models):
            print(f"  Getting predictions from model {i+1} (weight: {weights[i]:.4f})...")
            predictions = model.predict(data, verbose=0)
            all_predictions.append(predictions * weights[i])
        
        # Weighted sum
        weighted_predictions = np.sum(all_predictions, axis=0)
        
        return weighted_predictions
    
    def evaluate_ensemble(self, method='soft', weights=None):
        """
        Evaluate ensemble on validation set
        
        Args:
            method: Ensemble method ('soft', 'hard', 'weighted')
            weights: Weights for weighted average (optional)
            
        Returns:
            Accuracy and predictions
        """
        print("\n" + "="*70)
        print(f"ENSEMBLE EVALUATION - {method.upper()} VOTING")
        print("="*70)
        
        # Load data
        print("\nLoading validation data...")
        data_loader = RiceDiseaseDataLoader()
        generator = data_loader.create_validation_generator()
        
        print(f"✓ Loaded {generator.samples} validation samples")
        
        # Get true labels
        generator.reset()
        y_true = generator.classes
        
        # Get ensemble predictions
        if method == 'soft':
            y_pred_proba = self.predict_soft_voting(generator)
        elif method == 'hard':
            y_pred_proba = self.predict_hard_voting(generator)
        elif method == 'weighted':
            y_pred_proba = self.predict_weighted_average(generator, weights=weights)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        print("\n" + "="*70)
        print(f"ENSEMBLE ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("="*70)
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=config.CLASS_NAMES,
            digits=4
        )
        print("\nClassification Report:")
        print("-" * 70)
        print(report)
        
        # Save results
        results = {
            'method': method,
            'accuracy': float(accuracy),
            'num_models': len(self.models),
            'model_paths': self.model_paths,
            'weights': weights.tolist() if weights is not None else None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'classification_report': report
        }
        
        results_path = os.path.join(
            config.RESULTS_DIR, 
            f'ensemble_{method}_results.json'
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n✓ Results saved to: {results_path}")
        
        return accuracy, y_pred, y_pred_proba


def compare_ensemble_methods(model_paths):
    """
    Compare different ensemble methods
    
    Args:
        model_paths: List of paths to saved models
    """
    print("\n" + "="*70)
    print("COMPARING ENSEMBLE METHODS")
    print("="*70)
    
    ensemble = EnsemblePredictor(model_paths)
    
    results = []
    
    # Soft voting
    print("\n" + "="*70)
    print("[1/3] SOFT VOTING")
    print("="*70)
    acc_soft, _, _ = ensemble.evaluate_ensemble(method='soft')
    results.append(('Soft Voting', acc_soft))
    
    # Hard voting
    print("\n" + "="*70)
    print("[2/3] HARD VOTING")
    print("="*70)
    acc_hard, _, _ = ensemble.evaluate_ensemble(method='hard')
    results.append(('Hard Voting', acc_hard))
    
    # Weighted average (equal weights)
    print("\n" + "="*70)
    print("[3/3] WEIGHTED AVERAGE")
    print("="*70)
    acc_weighted, _, _ = ensemble.evaluate_ensemble(method='weighted')
    results.append(('Weighted Average', acc_weighted))
    
    # Compare individual models
    print("\n" + "="*70)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("="*70)
    
    data_loader = RiceDiseaseDataLoader()
    generator = data_loader.create_validation_generator()
    
    for i, model in enumerate(ensemble.models):
        generator.reset()
        y_true = generator.classes
        y_pred_proba = model.predict(generator, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        
        model_name = os.path.basename(model_paths[i])
        print(f"  Model {i+1} ({model_name}): {accuracy:.4f}")
        results.append((f'Model {i+1}', accuracy))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, acc in results:
        print(f"  {name:30s}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Find best method
    best_name, best_acc = max(results, key=lambda x: x[1])
    print(f"\n✓ Best Method: {best_name} with {best_acc:.4f} accuracy")
    print("="*70 + "\n")


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description='Ensemble Prediction for Rice Disease Detection')
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        help='Paths to saved model files (.h5)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='soft',
        choices=['soft', 'hard', 'weighted', 'compare'],
        help='Ensemble method (soft, hard, weighted, or compare all)'
    )
    parser.add_argument(
        '--weights',
        type=float,
        nargs='+',
        help='Weights for weighted average (must match number of models)'
    )
    
    args = parser.parse_args()
    
    if args.method == 'compare':
        # Compare all methods
        compare_ensemble_methods(args.models)
    else:
        # Use specific method
        ensemble = EnsemblePredictor(args.models)
        
        weights = None
        if args.weights:
            if len(args.weights) != len(args.models):
                print("Error: Number of weights must match number of models")
                return
            weights = np.array(args.weights)
        
        accuracy, y_pred, y_pred_proba = ensemble.evaluate_ensemble(
            method=args.method,
            weights=weights
        )
        
        print(f"\n✓ Ensemble evaluation completed!")
        print(f"✓ Check the '{config.RESULTS_DIR}' folder for detailed results\n")


if __name__ == "__main__":
    main()

