# Polymer Property Prediction with GNN

A Graph Neural Network solution for predicting polymer properties from SMILES strings.

## Quick Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Prepare your data:**
Place your competition data files in the same directory:
- `data/train.csv` (with columns: id, SMILES, Tg, FFV, Tc, Density, Rg)
- `data/test.csv` (with columns: id, SMILES)

## Usage

### Step 1: Preprocess Data
```bash
python data_preprocessing.py
```
This will:
- Convert SMILES strings to molecular graphs
- Extract molecular descriptors 
- Save processed graphs as PyTorch files

### Step 2: Train Model
```bash
python train.py
```
This will:
- Train a Graph Attention Network (GAT) 
- Use 80/20 train/validation split
- Save the best model as `best_model.pt`
- Generate training curves plot

### Step 3: Generate Predictions
```bash
python predict.py
```
This will:
- Load the trained model
- Generate predictions for test data
- Save submission file as `submission.csv`

## Files Overview

- **`data_preprocessing.py`** - Converts SMILES to molecular graphs with features
- **`gnn_model.py`** - Contains GNN model architectures (GAT and GCN)
- **`train.py`** - Training script with validation and model saving
- **`predict.py`** - Generates final predictions for submission
- **`utils.py`** - Helper functions for loss calculation and evaluation

## Model Architecture

The main model (`PolymerGNN`) uses:
- Graph Attention Network (GAT) layers with multi-head attention
- Molecular descriptor integration
- Multiple graph pooling methods (mean, max, sum)
- Multi-task learning for 5 polymer properties

## Key Features

- **Handles missing values** in training data
- **Custom weighted MAE loss** matching competition metric
- **Early stopping** to prevent overfitting
- **Residual connections** for better gradient flow
- **Molecular descriptors** combined with graph features
- **Attention mechanism** to focus on important molecular substructures

## Expected Results

- Training typically takes 50-100 epochs
- Model automatically saves best weights based on validation loss
- Final submission file will be generated as `submission.csv`

## Troubleshooting

1. **CUDA memory issues**: Reduce batch size in training
2. **RDKit errors**: Some SMILES might be invalid - these are automatically filtered
3. **Long training time**: Use the `SimpleGCN` model for faster experiments

## Customization

You can modify:
- Model architecture in `gnn_model.py`
- Training parameters in `train.py`
- Molecular features in `data_preprocessing.py`
- Loss function weights in `utils.py`