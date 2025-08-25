import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from gnn_model import PolymerGNN, SimpleGCN
from utils import WeightedMAELoss, calculate_competition_metric

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch)
        loss = criterion(out, batch.y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            
            all_preds.append(out.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    return total_loss / len(loader), all_preds, all_targets

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    train_graphs = torch.load('train_graphs.pt')
    
    # Filter out graphs with all NaN targets
    valid_graphs = []
    for graph in train_graphs:
        if not torch.isnan(graph.y).all():
            valid_graphs.append(graph)
    
    print(f"Using {len(valid_graphs)} valid training samples")
    
    # Split data
    train_data, val_data = train_test_split(valid_graphs, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    # Initialize model
    model = PolymerGNN(
        num_node_features=7,
        num_edge_features=3,
        global_features_dim=10,
        hidden_dim=128,
        num_layers=3,
        num_targets=5,
        dropout=0.2
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = WeightedMAELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.7)
    
    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        
        # Calculate competition metric
        comp_metric = calculate_competition_metric(val_preds, val_targets)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Competition Metric: {comp_metric:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'competition_metric': comp_metric
            }, 'best_model.pt')
            print("Saved best model!")
        
        print("-" * 50)
        
        # Early stopping
        if optimizer.param_groups[0]['lr'] < 1e-6:
            print("Learning rate too small, stopping training")
            break
    
    # Plot training curves
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Curves')
    
    plt.subplot(1, 2, 2)
    # Load best model for final evaluation
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, final_preds, final_targets = evaluate(model, val_loader, criterion, device)
    
    # Property-wise MAE
    property_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    maes = []
    
    for i in range(5):
        mask = ~np.isnan(final_targets[:, i])
        if mask.sum() > 0:
            mae = mean_absolute_error(final_targets[mask, i], final_preds[mask, i])
            maes.append(mae)
            print(f"{property_names[i]} MAE: {mae:.4f}")
        else:
            maes.append(0)
            print(f"{property_names[i]} MAE: No valid samples")
    
    plt.bar(property_names, maes)
    plt.ylabel('MAE')
    plt.title('Property-wise Performance')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Final competition metric: {checkpoint['competition_metric']:.4f}")

if __name__ == "__main__":
    main()