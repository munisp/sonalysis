# Sonalysis integration status

| Component | Current implementation | Remaining production validation |
|---|---|---|
| PostgreSQL | SQL schema, tenant/time indexes, compose service, optional DSN | HA/backups, RLS policy review, load test |
| TigerBeetle | Optional ledger profile in compose; ledger remains authoritative outside ML | Wire signed ledger client and reconciliation consumer |
| Redis | Optional runtime endpoint and compose service | Cache key policy, eviction/load test |
| Mojaloop | Optional profile with Postgres adapter variables; no MySQL dependency | Validate exact Mojaloop release configuration and DFSP certification |
| Kafka/Redpanda | Compose broker and API endpoint config | Add production ACLs, retention and consumer lag alerts |
| Fluvio | Optional profile | Select one event backbone per deployment; benchmark connector |
| APISIX | Compose gateway endpoint | Add declarative routes, OIDC plugin and rate limits |
| Keycloak | Compose identity provider | Realm/client provisioning and key rotation |
| OpenSearch | Compose search cluster | Index templates, shard sizing and security |
| Permify | Compose authorization service | Tenant relationship schema and policy tests |
| Temporal | Compose orchestration service | Register retraining workflow and retry policy |
| Lakehouse | Append-only partitioned JSONL event sink; training worker consumes lakehouse files | Replace JSONL with Parquet/object storage connector for production |
| AI/ML/DL | Real PyTorch MLP and graph encoder, trained weights, metrics, checkpoint promotion | Validate with real tenant labels; synthetic metrics are not production claims |
| CPU inference | Default `cpu`, `torch.inference_mode`, verified API latency ~6ms locally | Benchmark target hardware and batch/concurrency profile |

Sonalysis does not use a rules-only scorer. Rules remain limited to the decision threshold (`approve`, `review`, `decline`) over the learned probability.
