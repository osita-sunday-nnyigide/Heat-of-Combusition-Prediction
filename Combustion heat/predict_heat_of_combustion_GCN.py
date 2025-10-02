import os
import sys
import time
import numpy as np
import random
import pandas as pd
from scipy.stats import norm
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.data import InMemoryDataset

from sklearn.metrics import r2_score


RDLogger.DisableLog('rdApp.*')  
warnings.filterwarnings("ignore")

seed = 5 
random.seed(seed)              
np.random.seed(seed)           
torch.manual_seed(seed)       
torch.cuda.manual_seed(seed)  
torch.cuda.manual_seed_all(seed)  

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

start_time = time.time()

pauling_en = {
    1: 2.20,  6: 2.55,  7: 3.04,  8: 3.44,  9: 3.98,
    15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66
    
}



nmol=0
mw=0
def atom_features(atom):
    Z = atom.GetAtomicNum()
    return [
        float(Z),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(int(atom.GetIsAromatic())),
        float(atom.GetTotalNumHs()),
        float(atom.GetTotalValence()),
        float(atom.GetMass()),
        float(int(atom.GetHybridization())),     
        float(pauling_en.get(Z, 0.0)),            
        float(int(atom.IsInRingSize(3))),
        float(int(atom.IsInRingSize(4))),
        float(int(atom.IsInRingSize(5))),
        float(int(atom.IsInRingSize(6))),
        float(atom.GetImplicitValence())
    ]

def bond_features(bond):
    return [
        float(bond.GetBondTypeAsDouble()),
        float(int(bond.GetIsConjugated())),
        float(int(bond.IsInRing())),
        float(int(bond.GetStereo())),   
        float(int(bond.GetBondDir()))   
    ]

def get_descriptors(mol):

    return {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'RingCount': Descriptors.RingCount(mol),
        'MolMR': Descriptors.MolMR(mol),
        'ExactMolWt': Descriptors.ExactMolWt(mol),
        'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
        'NumRadicalElectrons': Descriptors.NumRadicalElectrons(mol),
        'BertzCT': Descriptors.BertzCT(mol)
        
    }

def featurize_molecule(mol):
    node_feats = np.array([atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32)
    edge_index = []
    edge_feats = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_feats.extend([feat, feat])

    edge_index = np.array(edge_index, dtype=np.int64).T 
    edge_feats = np.array(edge_feats, dtype=np.float32)
    
    return node_feats, edge_index, edge_feats


label_path = r"C:\Users\OS\Desktop\Combustion heat\heat_of_combution.csv"
label_df = pd.read_csv(label_path)

label_df["filename"] = label_df["filename"].astype(str)
label_df.set_index("filename", inplace=True)

sdf_dir = r"C:\Users\OS\Desktop\Combustion heat"
all_data = []
cid_in = []

for fname in os.listdir(sdf_dir):
    if not fname.endswith(".sdf"):
        continue
    
    sdf_path = os.path.join(sdf_dir, fname)
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = supplier[0] if supplier and supplier[0] is not None else None
    
    if mol is None:
        print(f"[Warning] Failed to load: {fname}")
        continue 

    try:
        cid = mol.GetProp("PUBCHEM_COMPOUND_CID")
    except KeyError:
        cid = os.path.splitext(fname)[0]
    
    cid_str = str(cid)
    node_feats, edge_idx, edge_feats = featurize_molecule(mol)
    descriptors = get_descriptors(mol)
    nmol+=1
    mw+=descriptors["MolWt"]

    if cid_str in label_df.index:
        labels = label_df.loc[cid_str].to_dict()
        cid_in.append(cid_str)
        all_data.append({
            'name': cid_str,
            'node_features': node_feats,
            'edge_index': edge_idx,
            'edge_features': edge_feats,
            'descriptors': descriptors,
            'labels': labels
        })
    else:
        print(f"[Info] No label found for CID: {cid_str}")


def to_pyg_data(entry):
    x = torch.tensor(entry['node_features'], dtype=torch.float)
    edge_index = torch.tensor(entry['edge_index'], dtype=torch.long)
    edge_attr = torch.tensor(entry['edge_features'], dtype=torch.float)

    descriptors = list(entry['descriptors'].values())
    desc_tensor = torch.tensor(descriptors, dtype=torch.float).unsqueeze(0)

    y = torch.tensor([entry['labels']['heat_of_combustion']], dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )
    data.descriptors = desc_tensor
    data.name = entry['name']
    return data

pyg_data_list = [to_pyg_data(entry) for entry in all_data]

all_y = torch.cat([data.y for data in pyg_data_list])
y_mean = all_y.mean()
y_std = all_y.std()

label_mean = y_mean
label_std = y_std

for data in pyg_data_list:
    data.y = (data.y - y_mean) / y_std


class ChemHazardDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__()
        self.data, self.slices = self.collate(data_list)

dataset = ChemHazardDataset(pyg_data_list)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


class ChemHazardGCN(nn.Module):
    def __init__(self, node_dim, desc_dim, hidden_dim, out_dim):
        super().__init__()

        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.fc_desc = nn.Linear(desc_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x, edge_index, batch, descriptors):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = global_mean_pool(x, batch)
        d = F.relu(self.fc_desc(descriptors))

        out = torch.cat([x, d], dim=1)

        return self.fc_out(out)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ChemHazardGCN(
    node_dim=dataset[0].x.shape[1],
    desc_dim=dataset[0].descriptors.shape[1],
    hidden_dim=16,
    out_dim=1
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
loss_fn = nn.MSELoss() 

epochs_list = []

train_loss_norm_list = []
train_loss_denorm_list = []

train_mae_list = []
test_mae_list = []

best_test_mae = float('inf')  
epochs_no_improve = 0         
patience = 4                  

for epoch in range(200):
    # ----- Training -----
    model.train()
    total_loss, total_train_mae, total_loss_denorm, total_train_mae_datawise, total_test_mae_datawise, total_train_samples,  total_test_samples = 0, 0, 0, 0, 0, 0, 0
    for batch in train_loader:
        batch = batch.to(device)
        descriptors = torch.cat([data.descriptors for data in batch.to_data_list()], dim=0).to(device)

        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch, descriptors)
        loss = loss_fn(out.squeeze(), batch.y.squeeze())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        out_denorm = out.squeeze() * label_std + label_mean
        y_denorm = batch.y.squeeze() * label_std + label_mean
        total_train_mae += torch.mean(torch.abs(out_denorm - y_denorm)).item()

        total_train_mae_datawise += torch.sum(torch.abs(out_denorm - y_denorm)).item()
        total_train_samples += y_denorm.size(0)

        loss_denorm = loss_fn(out_denorm, y_denorm)
        total_loss_denorm += loss_denorm.item()
    

    avg_train_loss = total_loss / len(train_loader)
    avg_train_loss_denorm = total_loss_denorm / len(train_loader)
    avg_train_mae = total_train_mae / len(train_loader)
    avg_train_mae_datawise = total_train_mae_datawise / total_train_samples

    train_mae_list.append(avg_train_mae)


    epochs_list.append(epoch + 1)
    train_loss_norm_list.append(avg_train_loss)
    train_loss_denorm_list.append(avg_train_loss_denorm)

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            descriptors = torch.cat([data.descriptors for data in batch.to_data_list()], dim=0).to(device)
            out = model(batch.x, batch.edge_index, batch.batch, descriptors)
            
            out_denorm = out.squeeze().cpu() * label_std + label_mean
            y_denorm = batch.y.squeeze().cpu() * label_std + label_mean

            total_test_mae_datawise += torch.sum(torch.abs(out_denorm - y_denorm)).item()
            total_test_samples += y_denorm.size(0)

            preds.append(out_denorm)
            targets.append(y_denorm)


    preds = torch.cat(preds)
    targets = torch.cat(targets)
    test_mae = torch.mean(torch.abs(preds - targets)).item()
    avg_test_mae_datawise = total_test_mae_datawise / total_test_samples

    test_mae_list.append(test_mae)

    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss_denorm:.4f} | " f"Train MAE: {avg_train_mae:.4f} | " 
         f"Data-wise Train MAE: {avg_train_mae_datawise:.4f} | Test MAE: {test_mae:.4f} | Data-wise Test MAE: {avg_test_mae_datawise:.4f}")

    if test_mae < best_test_mae:  
        best_test_mae = test_mae
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model.pth")  
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(torch.load("best_model.pth")) 
            print("duration = ", (time.time()-start_time)/60)
            break



y_mean = torch.mean(targets)

sst = torch.sum((targets - y_mean) ** 2)

sse = torch.sum((targets - preds) ** 2)

r2 = 1 - sse / sst
print("R^2:", r2.item())


