from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from .config import get_settings
from .schemas import ScoreRequest, ScoreResponse, ModelInfo
from .artifacts import load_artifact, metadata
from .synthetic import frame_to_features
from .events import append_score_event

settings=get_settings(); app=FastAPI(title='Sonalysis Scoring API',version='0.1.0')
SCORES=Counter('sonalysis_scores_total','Scores by decision',['tenant','decision']); LATENCY=Histogram('sonalysis_score_latency_seconds','Score latency')
model=None; ckpt=None

def _load():
    global model,ckpt
    active=settings.model_dir/'active.json'
    if not active.exists(): return
    spec=json.loads(active.read_text()); model,ckpt=load_artifact(Path(spec['path']),settings.device)
_load()

def _frame(req):
    t=req.transaction
    return pd.DataFrame([{'amount':t.amount,'account_age_days':t.account_age_days,'is_new_device':int(t.is_new_device),'failed_auth_count_24h':t.failed_auth_count_24h,'velocity_1h':t.velocity_1h,'prior_chargebacks':t.prior_chargebacks,'graph_degree':t.graph_degree,'graph_fraud_neighbor_rate':t.graph_fraud_neighbor_rate,'hour':t.timestamp.hour,'distance_from_home':0.0,'merchant_risk':0.20,'device_reuse':0,'channel_risk':0.15}])

@app.get('/healthz')
def health(): return {'status':'ok','model_loaded':model is not None,'cpu_inference':True,'integrations':{'postgres':bool(settings.postgres_dsn),'redis':bool(settings.redis_url),'kafka':bool(settings.kafka_bootstrap)}}
@app.get('/readyz')
def ready():
    if model is None: raise HTTPException(503,'No trained model artifact; run sonalysis-train')
    return {'status':'ready','model':ckpt['version']}
@app.get('/v1/model',response_model=ModelInfo)
def model_info():
    if not ckpt: raise HTTPException(503,'model unavailable')
    return metadata(ckpt)
@app.get('/metrics')
def metrics(): return Response(generate_latest(),media_type=CONTENT_TYPE_LATEST)
@app.post('/v1/score',response_model=ScoreResponse)
def score(req: ScoreRequest):
    if model is None: raise HTTPException(503,'model unavailable')
    start=time.perf_counter(); x=frame_to_features(_frame(req)); x=(x-ckpt['mean'])/(ckpt['std']);
    with torch.inference_mode(): probability=float(torch.sigmoid(model(torch.tensor(x,dtype=torch.float32))).item())
    decision='decline' if probability>=.80 else ('review' if probability>=.45 else 'approve'); elapsed=(time.perf_counter()-start)*1000
    SCORES.labels(req.transaction.tenant_id,decision).inc(); LATENCY.observe(elapsed/1000)
    append_score_event(settings.lakehouse_path, {'event_type':'score','transaction_id':req.transaction.transaction_id,'tenant_id':req.transaction.tenant_id,'model_version':ckpt['version'],'fraud_probability':probability,'decision':decision,'latency_ms':elapsed,'occurred_at':req.transaction.timestamp.isoformat()})
    explanation={'signals':[]}
    if req.explain:
        t=req.transaction
        for name,val in [('new_device',t.is_new_device),('velocity_1h',t.velocity_1h),('failed_auth_24h',t.failed_auth_count_24h),('fraud_neighbor_rate',t.graph_fraud_neighbor_rate)]:
            if val: explanation['signals'].append({'feature':name,'value':val})
    return ScoreResponse(transaction_id=req.transaction.transaction_id,tenant_id=req.transaction.tenant_id,fraud_probability=probability,decision=decision,model_version=ckpt['version'],latency_ms=elapsed,explanation=explanation)
def run():
    import uvicorn; uvicorn.run(app,host='0.0.0.0',port=8080)
