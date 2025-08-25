import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import pickle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def smiles_to_graph(smiles):
    """Convert SMILES string to PyTorch Geometric graph"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Node features (atoms)
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().real,
            atom.GetIsAromatic(),
            atom.GetMass(),
            atom.GetTotalNumHs(),
        ]
        atom_features.append(features)
    
    # Edge indices and features (bonds)
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Add both directions for undirected graph
        edge_indices.extend([[i, j], [j, i]])
        
        bond_feature = [
            bond.GetBondTypeAsDouble(),
            bond.GetIsConjugated(),
            bond.GetIsInRing(),
        ]
        edge_features.extend([bond_feature, bond_feature])
    
    # Convert to tensors
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def get_molecular_descriptors(smiles):
    """Extract molecular descriptors from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * 10
    
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumSaturatedRings(mol),
        Descriptors.TPSA(mol),
        Descriptors.BertzCT(mol),
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
    ]
    
    return descriptors

def preprocess_data():
    """Main preprocessing function"""
    print("Loading data...")
    train_df = pd.read_csv('dataset-neurips-open-polymer-prediction-2025/train.csv')
    test_df = pd.read_csv('dataset-neurips-open-polymer-prediction-2025/test.csv')
    
    # Target columns
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    print("Converting SMILES to graphs...")
    
    # Process training data
    train_graphs = []
    train_descriptors = []
    train_targets = []
    
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing train"):
        smiles = row['SMILES']
        
        # Convert to graph
        graph = smiles_to_graph(smiles)
        if graph is None:
            continue
            
        # Get molecular descriptors
        descriptors = get_molecular_descriptors(smiles)
        
        # Get targets (handle missing values)
        targets = []
        for col in target_cols:
            if pd.isna(row[col]):
                targets.append(float('nan'))
            else:
                targets.append(row[col])
        
        train_graphs.append(graph)
        train_descriptors.append(descriptors)
        train_targets.append(targets)
    
    # Process test data
    test_graphs = []
    test_descriptors = []
    test_ids = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing test"):
        smiles = row['SMILES']
        
        graph = smiles_to_graph(smiles)
        if graph is None:
            continue
            
        descriptors = get_molecular_descriptors(smiles)
        
        test_graphs.append(graph)
        test_descriptors.append(descriptors)
        test_ids.append(row['id'])
    
    print("Normalizing descriptors...")
    
    # Normalize molecular descriptors
    all_descriptors = train_descriptors + test_descriptors
    scaler = StandardScaler()
    all_descriptors_scaled = scaler.fit_transform(all_descriptors)
    
    train_descriptors_scaled = all_descriptors_scaled[:len(train_descriptors)]
    test_descriptors_scaled = all_descriptors_scaled[len(train_descriptors):]
    
    # Add descriptors to graphs
    for i, graph in enumerate(train_graphs):
        graph.global_features = torch.tensor(train_descriptors_scaled[i], dtype=torch.float)
        graph.y = torch.tensor(train_targets[i], dtype=torch.float)
        
    for i, graph in enumerate(test_graphs):
        graph.global_features = torch.tensor(test_descriptors_scaled[i], dtype=torch.float)
    
    print("Saving processed data...")
    
    # Save processed data
    torch.save(train_graphs, 'train_graphs.pt')
    torch.save(test_graphs, 'test_graphs.pt')
    torch.save(test_ids, 'test_ids.pt')
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Processed {len(train_graphs)} training samples and {len(test_graphs)} test samples")
    print("Preprocessing complete!")

if __name__ == "__main__":
    preprocess_data()