from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from .models import TransactionMLP, GraphFraudEncoder
from .synthetic import FEATURES

def save_artifact(path: Path, model, mean, std, metrics, model_type='transaction_mlp', version='transaction-mlp-v1'):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'mean': np.asarray(mean, dtype='float32'), 'std': np.asarray(std, dtype='float32'), 'metrics': metrics, 'model_type': model_type, 'version': version, 'features': FEATURES}, path)

def load_artifact(path: Path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cls = GraphFraudEncoder if ckpt['model_type'] == 'graph_gnn' else TransactionMLP
    model = cls(len(ckpt['features']))
    model.load_state_dict(ckpt['state_dict']); model.to(device).eval()
    return model, ckpt

def metadata(ckpt):
    return {'version': ckpt['version'], 'model_type': ckpt['model_type'], 'trained_at': ckpt.get('trained_at','unknown'), 'device': 'cpu', 'metrics': ckpt.get('metrics',{}), 'feature_schema': ckpt['features']}
