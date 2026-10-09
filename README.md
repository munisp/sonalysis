# Sonalysis Platform

Sonalysis is an end-to-end, multi-tenant fraud scoring platform. This repository contains **working code**, not a rules-only demo:

- PyTorch transaction MLP with shipped, reproducibly trained CPU weights.
- PyTorch relational graph encoder using transaction-derived node features and deterministic neighbor aggregation (no torch-geometric dependency).
- Synthetic Nigerian-style transaction generator with tenant, POS, device, merchant, geo, time, velocity and fraud-ring patterns.
- Training, validation, calibration, checkpointing, resume, and fine-tuning scripts.
- FastAPI inference API with tenant isolation, model version selection, health/metrics, and optional Postgres/Redis/Kafka integrations.
- Postgres schema for transactions, labels, predictions, model registry and drift windows.
- Lakehouse export contract (Parquet-compatible event batches) and continuous-training worker.
- Docker Compose integration profiles for Postgres, Redis, Kafka/Redpanda, OpenSearch, APISIX, Keycloak, Permify, Temporal, Dapr, Fluvio, TigerBeetle and Mojaloop.

## Honest status

The generated checkpoint in `artifacts/` is trained on synthetic data and is **not evidence of production accuracy**. Real-world launch requires tenant-approved labels, replay evaluation, calibration, privacy review and monitoring thresholds. The API falls back to a deterministic baseline only when model artifacts are unavailable; normal startup requires a trained checkpoint.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
python -m sonalysis.training --samples 50000 --epochs 8 --output artifacts
uvicorn sonalysis.api:app --app-dir src --host 0.0.0.0 --port 8080
curl -X POST http://localhost:8080/v1/score -H 'content-type: application/json' -d @examples/score.json
```

CPU inference is supported and is the default. GPU is used only when explicitly configured.

## Continuous training

`python -m sonalysis.continuous_training` consumes labeled events from the lakehouse export directory, applies tenant-aware time splits, fine-tunes from the active checkpoint, evaluates holdout metrics, and promotes only if the configured PR-AUC and calibration gates pass.
