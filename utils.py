import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error

class WeightedMAELoss(nn.Module):
    """
    Implements the competition's weighted MAE loss function
    """
    def __init__(self):
        super(WeightedMAELoss, self).__init__()
        self.property_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
    def forward(self, predictions, targets):
        """
        Calculate weighted MAE loss
        
        Args:
            predictions: Model predictions [batch_size, 5]
            targets: True values [batch_size, 5]
        """
        batch_size, num_props = predictions.shape
        total_loss = 0
        valid_props = 0
        
        for prop_idx in range(num_props):
            # Get valid (non-NaN) samples for this property
            mask = ~torch.isnan(targets[:, prop_idx])
            
            if mask.sum() == 0:
                continue  # Skip if no valid samples
                
            pred_prop = predictions[mask, prop_idx]
            target_prop = targets[mask, prop_idx]
            
            # Calculate MAE for this property
            mae = torch.mean(torch.abs(pred_prop - target_prop))
            total_loss += mae
            valid_props += 1
        
        return total_loss / max(valid_props, 1)

def calculate_competition_metric(predictions, targets):
    """
    Calculate the exact competition metric (weighted MAE)
    
    Args:
        predictions: numpy array [n_samples, 5]
        targets: numpy array [n_samples, 5]
    """
    property_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    num_props = len(property_names)
    
    # Calculate per-property statistics
    property_stats = {}
    total_weighted_error = 0
    sum_weights = 0
    
    for prop_idx in range(num_props):
        # Get valid samples for this property
        mask = ~np.isnan(targets[:, prop_idx])
        
        if mask.sum() == 0:
            continue
            
        pred_prop = predictions[mask, prop_idx]
        target_prop = targets[mask, prop_idx]
        
        # Calculate MAE
        mae = mean_absolute_error(target_prop, pred_prop)
        
        # Calculate property range (approximation)
        prop_range = np.ptp(target_prop)  # peak-to-peak range
        if prop_range == 0:
            prop_range = 1  # Avoid division by zero
            
        # Calculate number of samples
        n_samples = mask.sum()
        
        # Calculate weight (simplified version of competition formula)
        weight = 1.0 / (prop_range * np.sqrt(n_samples))
        
        property_stats[property_names[prop_idx]] = {
            'mae': mae,
            'n_samples': n_samples,
            'range': prop_range,
            'weight': weight
        }
        
        total_weighted_error += weight * mae
        sum_weights += weight
    
    # Normalize weights
    if sum_weights > 0:
        weighted_mae = (total_weighted_error / sum_weights) * num_props
    else:
        weighted_mae = 0
        
    return weighted_mae

def print_model_summary(model):
    """Print model architecture and parameter count"""
    print("Model Architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

def save_predictions_with_stats(predictions, test_ids, filename='predictions_with_stats.csv'):
    """
    Save predictions with detailed statistics
    
    Args:
        predictions: numpy array [n_samples, 5]
        test_ids: list of test sample IDs
        filename: output filename
    """
    import pandas as pd
    
    property_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Create DataFrame
    df = pd.DataFrame({
        'id': test_ids,
        'Tg': predictions[:, 0],
        'FFV': predictions[:, 1], 
        'Tc': predictions[:, 2],
        'Density': predictions[:, 3],
        'Rg': predictions[:, 4]
    })
    
    # Print statistics
    print("\nPrediction Statistics:")
    print("-" * 60)
    for i, prop in enumerate(property_names):
        values = predictions[:, i]
        print(f"{prop:8s}: Mean={values.mean():8.4f}, Std={values.std():8.4f}")
        print(f"         Min={values.min():8.4f}, Max={values.max():8.4f}")
        print(f"         Q25={np.percentile(values, 25):8.4f}, Q75={np.percentile(values, 75):8.4f}")
        print("-" * 60)
    
    # Save to file
    df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

def validate_predictions(predictions):
    """
    Validate predictions for common issues
    
    Args:
        predictions: numpy array [n_samples, 5]
    """
    property_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    print("Validation Report:")
    print("-" * 40)
    
    for i, prop in enumerate(property_names):
        values = predictions[:, i]
        
        # Check for NaN or infinite values
        nan_count = np.isnan(values).sum()
        inf_count = np.isinf(values).sum()
        
        # Check for extreme values (might indicate issues)
        extreme_low = (values < -1000).sum()
        extreme_high = (values > 1000).sum()
        
        print(f"{prop}:")
        print(f"  NaN values: {nan_count}")
        print(f"  Inf values: {inf_count}")
        print(f"  Extreme low (<-1000): {extreme_low}")
        print(f"  Extreme high (>1000): {extreme_high}")
        
        if nan_count > 0 or inf_count > 0:
            print(f"  WARNING: Invalid values detected!")
        
        print()

def create_ensemble_predictions(predictions_list, weights=None):
    """
    Create ensemble predictions from multiple models
    
    Args:
        predictions_list: List of prediction arrays
        weights: Optional weights for each model
    """
    if weights is None:
        weights = [1.0] * len(predictions_list)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Weighted average
    ensemble_preds = np.zeros_like(predictions_list[0])
    
    for preds, weight in zip(predictions_list, weights):
        ensemble_preds += weight * preds
    
    return ensemble_preds

class EarlyStopping:
    """
    Early stopping utility to prevent overfitting
    """
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def plot_learning_curves(train_losses, val_losses, save_path='learning_curves.png'):
    """
    Plot and save learning curves
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_predictions_by_property(predictions, targets, property_names=None):
    """
    Analyze predictions for each property separately
    """
    if property_names is None:
        property_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    results = {}
    
    for i, prop in enumerate(property_names):
        # Get valid samples
        mask = ~np.isnan(targets[:, i])
        
        if mask.sum() == 0:
            results[prop] = {'mae': None, 'n_samples': 0}
            continue
            
        pred_prop = predictions[mask, i]
        target_prop = targets[mask, i]
        
        # Calculate metrics
        mae = mean_absolute_error(target_prop, pred_prop)
        
        # Calculate correlation
        correlation = np.corrcoef(pred_prop, target_prop)[0, 1]
        
        # Calculate R²
        ss_res = np.sum((target_prop - pred_prop) ** 2)
        ss_tot = np.sum((target_prop - np.mean(target_prop)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        results[prop] = {
            'mae': mae,
            'correlation': correlation,
            'r2': r2,
            'n_samples': mask.sum(),
            'pred_mean': pred_prop.mean(),
            'pred_std': pred_prop.std(),
            'target_mean': target_prop.mean(),
            'target_std': target_prop.std()
        }
    
    return results