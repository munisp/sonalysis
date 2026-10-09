from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve
from .synthetic import make_dataset, frame_to_features
from .models import TransactionMLP, GraphFraudEncoder
from .artifacts import save_artifact

def train(samples=50000, epochs=8, batch_size=1024, seed=7, output=Path('artifacts'), model_type='transaction_mlp', resume=None, lr=2e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    df = make_dataset(samples, seed); df = df.sort_values('timestamp').reset_index(drop=True)
    cut = int(len(df)*.8); x = frame_to_features(df); y = df.label.to_numpy('float32')
    mean, std = x[:cut].mean(0), x[:cut].std(0)+1e-6
    x = (x-mean)/std
    device = torch.device('cuda' if torch.cuda.is_available() and __import__('os').getenv('SONALYSIS_DEVICE')=='cuda' else 'cpu')
    model = (GraphFraudEncoder if model_type=='graph_gnn' else TransactionMLP)(x.shape[1]).to(device)
    if resume:
        ckpt=torch.load(resume,map_location=device,weights_only=False); model.load_state_dict(ckpt['state_dict']); mean,std=ckpt['mean'],ckpt['std']
    train_loader=DataLoader(TensorDataset(torch.tensor(x[:cut]),torch.tensor(y[:cut])),batch_size=batch_size,shuffle=True)
    pos=max(y[:cut].sum(),1); neg=max(len(y[:cut])-pos,1)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],device=device)); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    for epoch in range(epochs):
        model.train(); running=0.0
        for xb,yb in train_loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2); opt.step(); running += loss.item()*len(xb)
        model.eval()
        with torch.no_grad(): p=torch.sigmoid(model(torch.tensor(x[cut:]).to(device))).cpu().numpy()
        metrics={'roc_auc':float(roc_auc_score(y[cut:],p)),'pr_auc':float(average_precision_score(y[cut:],p)),'brier':float(brier_score_loss(y[cut:],p)),'val_loss':float(running/cut)}
        print(json.dumps({'epoch':epoch+1,'metrics':metrics}))
    version=f"{'gnn' if model_type=='graph_gnn' else 'transaction-mlp'}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    path=output/f'{version}.pt'; save_artifact(path,model,mean,std,metrics,model_type,version)
    (output/'active.json').write_text(json.dumps({'path':str(path),'version':version,'metrics':metrics},indent=2))
    return path,metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument('--samples',type=int,default=50000); p.add_argument('--epochs',type=int,default=8); p.add_argument('--batch-size',type=int,default=1024); p.add_argument('--output',type=Path,default=Path('artifacts')); p.add_argument('--model-type',choices=['transaction_mlp','graph_gnn'],default='transaction_mlp'); p.add_argument('--resume'); p.add_argument('--lr',type=float,default=2e-3)
    a=p.parse_args(); train(a.samples,a.epochs,a.batch_size,output=a.output,model_type=a.model_type,resume=a.resume,lr=a.lr)
if __name__=='__main__': main()
