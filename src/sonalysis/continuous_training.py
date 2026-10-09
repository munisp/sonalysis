from __future__ import annotations
import argparse, json
from pathlib import Path
from .training import train
from .config import get_settings

def main():
    p=argparse.ArgumentParser(); p.add_argument('--lakehouse',type=Path,default=None); p.add_argument('--epochs',type=int,default=2); p.add_argument('--samples',type=int,default=20000); a=p.parse_args(); s=get_settings()
    lake=a.lakehouse or s.lakehouse_path
    # Event files are intentionally treated as an append-only lakehouse contract. Synthetic fallback keeps CI reproducible.
    event_count=sum(1 for _ in lake.glob('**/*.jsonl')) if lake.exists() else 0
    path,metrics=train(samples=max(a.samples,event_count*100),epochs=a.epochs,output=s.model_dir,resume=None,lr=5e-4)
    if metrics['pr_auc'] < s.min_promote_prauc or metrics['brier'] > s.max_promote_brier:
        path.unlink(missing_ok=True); raise SystemExit(f'model rejected: {metrics}')
    print(json.dumps({'promoted':str(path),'metrics':metrics,'event_files':event_count}))
if __name__=='__main__': main()
