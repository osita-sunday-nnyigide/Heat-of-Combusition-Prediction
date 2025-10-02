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
import matplotlib.ticker as ticker
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.data import InMemoryDataset

from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split


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

dataset = [to_pyg_data(entry) for entry in all_data]

X = np.vstack([d.descriptors.numpy() for d in dataset])
y = np.hstack([d.y.numpy() for d in dataset])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "reg:squarederror",
    "eval_metric": ["mae", "rmse"],
    "eta": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "gpu_hist" if torch.cuda.is_available() else "hist",
    "seed": 42
}

evals_result = {}
model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, "train"), (dtest, "test")],
    evals_result=evals_result,
    early_stopping_rounds=4,
    verbose_eval=True,
    maximize=False,  
)

print("duration = ", (time.time()-start_time)/60)

y_pred = model.predict(dtest)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

