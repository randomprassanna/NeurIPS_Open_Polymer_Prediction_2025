import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from gnn_model import PolymerGNN
from tqdm import tqdm

def predict(model, loader, device):
    """Generate predictions for test data"""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            batch = batch.to(device)
            out = model(batch)
            all_preds.append(out.cpu().numpy())
    
    return np.vstack(all_preds)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    print("Loading test data...")
    test_graphs = torch.load('test_graphs.pt', weights_only=False)
    test_ids = torch.load('test_ids.pt', weights_only=False)
    
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    # Load trained model
    print("Loading trained model...")
    model = PolymerGNN(
        num_node_features=7,
        num_edge_features=3,
        global_features_dim=10,
        hidden_dim=128,
        num_layers=3,
        num_targets=5,
        dropout=0.2
    ).to(device)
    
    checkpoint = torch.load('best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss: {checkpoint['val_loss']:.4f}")
    
    # Generate predictions
    print("Generating predictions...")
    predictions = predict(model, test_loader, device)
    
    # Create submission file
    property_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    submission_df = pd.DataFrame({
        'id': test_ids,
        'Tg': predictions[:, 0],
        'FFV': predictions[:, 1],
        'Tc': predictions[:, 2],
        'Density': predictions[:, 3],
        'Rg': predictions[:, 4]
    })
    
    # Handle any NaN predictions
    submission_df = submission_df.fillna(0.0)
    
    # Save submission
    submission_df.to_csv('submission.csv', index=False)
    
    print("Submission saved to 'submission.csv'")
    print(f"Generated predictions for {len(test_ids)} test samples")
    
    # Display prediction statistics
    print("\nPrediction Statistics:")
    for i, prop in enumerate(property_names):
        pred_values = predictions[:, i]
        print(f"{prop}: Mean={pred_values.mean():.4f}, Std={pred_values.std():.4f}, "
              f"Min={pred_values.min():.4f}, Max={pred_values.max():.4f}")

if __name__ == "__main__":
    main()