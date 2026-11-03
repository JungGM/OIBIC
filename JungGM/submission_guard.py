"""Keep accepted/submitted official artifacts immutable during later experiments."""
import json
from pathlib import Path


def ensure_writable(directory):
    directory=Path(directory)
    record=directory/'validation.json'
    if record.exists() and json.loads(record.read_text(encoding='utf-8-sig')).get('official_submitted'):
        raise ValueError('Officially submitted candidate is immutable; use a new candidate folder')
    if any(directory.glob('공식제출결과*')):
        raise ValueError('Official submission receipt exists; refuse to overwrite candidate')
