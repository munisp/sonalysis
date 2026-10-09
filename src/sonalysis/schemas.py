from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field, ConfigDict

class Transaction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    transaction_id: str
    tenant_id: str
    amount: float = Field(gt=0)
    currency: str = 'NGN'
    merchant_id: str
    merchant_category: str = 'general'
    customer_id: str
    device_id: str
    channel: str = 'pos'
    country: str = 'NG'
    latitude: float = 9.08
    longitude: float = 7.40
    timestamp: datetime
    account_age_days: int = Field(default=365, ge=0)
    is_new_device: bool = False
    failed_auth_count_24h: int = Field(default=0, ge=0)
    velocity_1h: int = Field(default=1, ge=0)
    prior_chargebacks: int = Field(default=0, ge=0)
    graph_degree: int = Field(default=1, ge=0)
    graph_fraud_neighbor_rate: float = Field(default=0.0, ge=0, le=1)

class ScoreRequest(BaseModel):
    transaction: Transaction
    model_version: str | None = None
    explain: bool = True

class ScoreResponse(BaseModel):
    transaction_id: str
    tenant_id: str
    fraud_probability: float
    decision: str
    model_version: str
    latency_ms: float
    explanation: dict[str, Any] = {}

class ModelInfo(BaseModel):
    version: str
    model_type: str
    trained_at: str
    device: str
    metrics: dict[str, float]
    feature_schema: list[str]
