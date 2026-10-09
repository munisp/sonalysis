from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

_lock = Lock()

def append_score_event(root: Path, event: dict) -> None:
    """Persist an auditable JSONL event; partitions are UTC day and tenant."""
    tenant = str(event['tenant_id']).replace('/', '_')
    day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    path = root / f'date={day}' / f'tenant={tenant}' / 'scores.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(event, separators=(',', ':'), default=str) + '\n')
