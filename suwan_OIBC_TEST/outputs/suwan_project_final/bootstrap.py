"""Download the attributed baseline at a pinned commit; never overwrite a local run."""
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

HERE=Path(__file__).resolve().parent
DEST=HERE.parent/'suwan_projectV2'

def main():
    if DEST.exists() and any(DEST.iterdir()):
        raise SystemExit('Baseline directory already exists; keep it unchanged. Bootstrap is for a fresh checkout.')
    inventory=json.loads((HERE/'UPSTREAM.json').read_text(encoding='utf-8-sig'))
    DEST.mkdir(parents=True,exist_ok=True)
    original=DEST/'upstream';original.mkdir(exist_ok=True)
    for item in inventory['files']:
        name=item['name']
        url=f"https://raw.githubusercontent.com/{inventory['repo']}/{inventory['commit']}/JungGM/{name}"
        with urllib.request.urlopen(url,timeout=60) as response:data=response.read()
        sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if sha!=item['blob_sha']:raise ValueError('Upstream blob mismatch: '+name)
        (original/name).write_bytes(data)
        if name.endswith('.py'):
            source=data.decode('utf-8-sig')
            if name=='pipeline.py':
                source=source.replace('THREADS = 6','THREADS = 2').replace('folder.mkdir()','folder.mkdir(exist_ok=True)')
            (DEST/name).write_text(source,encoding='utf-8')
    shutil.copyfile(HERE/'baseline_runner.py',DEST/'run_reproduction.py')
    shutil.copyfile(HERE/'UPSTREAM.json',DEST/'UPSTREAM.json')
    audit=(DEST/'audit_run.py').read_text(encoding='utf-8')
    audit=audit.replace("model_meta = list((run/'production'/'models').glob('*.json'))",
                        "model_meta = [f for f in (run/'production'/'models').glob('*.json') if not f.name.endswith('_checkpoint.json')]")
    (DEST/'audit_compatible.py').write_text(audit,encoding='utf-8')
    print('Pinned baseline downloaded with Git blob hash verification:',DEST)

if __name__=='__main__':main()
