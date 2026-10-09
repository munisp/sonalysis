from __future__ import annotations
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

CATEGORIES = ['grocery','electronics','airtime','pharmacy','fuel','fashion','general']
CHANNELS = ['pos','ussd','mobile','web','agent']
FEATURES = ['log_amount','account_age_days','is_new_device','failed_auth_count_24h','velocity_1h','prior_chargebacks','graph_degree','graph_fraud_neighbor_rate','hour_sin','hour_cos','distance_from_home','merchant_risk','device_reuse','channel_risk']

def make_dataset(n: int = 50_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenants = rng.choice(['tenant-ng-1','tenant-ng-2','tenant-gh-1','tenant-ke-1'], n, p=[.55,.2,.15,.1])
    fraud_ring = rng.binomial(1, .035, n)
    amount = np.exp(rng.normal(np.log(6500), 1.05, n)).clip(100, 2_000_000)
    account_age = rng.exponential(420, n).clip(1, 5000).astype(int)
    new_device = rng.binomial(1, .10 + .30*fraud_ring, n)
    failed = rng.poisson(.25 + 1.8*fraud_ring, n)
    velocity = rng.poisson(2 + 9*fraud_ring, n)
    chargebacks = rng.poisson(.06 + .8*fraud_ring, n)
    degree = rng.poisson(3 + 10*fraud_ring, n)
    neighbor_rate = np.clip(rng.beta(1.2 + 4*fraud_ring, 15 - 8*fraud_ring), 0, 1)
    hour = rng.integers(0, 24, n)
    dist = np.abs(rng.normal(8 + 18*fraud_ring, 12, n))
    merchant_risk = rng.beta(2 + 2*fraud_ring, 7, n)
    device_reuse = rng.poisson(1 + 5*fraud_ring, n)
    channel = rng.choice(CHANNELS, n, p=[.45,.15,.2,.1,.1])
    channel_risk = np.select([channel=='web', channel=='agent', channel=='ussd'], [.50,.35,.25], .15)
    logit = (-5.0 + .0000009*amount + .55*new_device + .20*failed + .16*velocity + .75*chargebacks + .11*degree + 2.1*neighbor_rate + .045*dist + 1.4*merchant_risk + .22*device_reuse + channel_risk + 1.8*fraud_ring)
    prob = 1/(1+np.exp(-logit))
    label = rng.binomial(1, np.clip(prob, .001, .98))
    ts = [datetime.now(timezone.utc) - timedelta(minutes=int(x)) for x in rng.integers(0, 60*24*180, n)]
    return pd.DataFrame({'tenant_id':tenants,'amount':amount,'account_age_days':account_age,'is_new_device':new_device,'failed_auth_count_24h':failed,'velocity_1h':velocity,'prior_chargebacks':chargebacks,'graph_degree':degree,'graph_fraud_neighbor_rate':neighbor_rate,'hour':hour,'distance_from_home':dist,'merchant_risk':merchant_risk,'device_reuse':device_reuse,'channel_risk':channel_risk,'timestamp':ts,'label':label})

def frame_to_features(df: pd.DataFrame) -> np.ndarray:
    out = df.copy()
    out['log_amount'] = np.log1p(out['amount'])
    out['hour_sin'] = np.sin(2*np.pi*out['hour']/24)
    out['hour_cos'] = np.cos(2*np.pi*out['hour']/24)
    return out[FEATURES].astype('float32').to_numpy()
