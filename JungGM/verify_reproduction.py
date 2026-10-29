"""Verify byte-identical submission recreation from serialized final models."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(run, submission, reproduction):
    completed = read(run/'completion.json')
    recreated = read(reproduction/'receipt.json')
    if recreated.get('refitted') is not False or recreated.get('inference_only') is not True:
        raise ValueError('Expected inference-only reproduction')
    if Path(recreated['run']).resolve() != run.resolve():
        raise ValueError('Reproduction belongs to a different run')
    matches = {}
    for name in ('composition.csv', 'forecast.csv'):
        original, repeated = digest(submission/name), digest(reproduction/name)
        if len({original, repeated, completed['files'][name], recreated['files'][name]}) != 1:
            raise ValueError('Reproduction hash mismatch: '+name)
        matches[name] = original
    for name, expected in recreated['model_files_sha256'].items():
        if digest(run/name) != expected:
            raise ValueError('Serialized model changed after reproduction: '+name)
    result = {'passed': True, 'byte_identical': True, 'refitted': False,
        'files_sha256': matches, 'model_files_verified': len(recreated['model_files_sha256'])}
    (run/'reproduction_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('PASS: both submission CSVs are byte-identical after saved-model inference')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'submission', 'reproduction'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    verify(args.run, args.submission, args.reproduction)
