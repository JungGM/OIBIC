"""Run the documented finite experiment from the user's local competition data."""
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'outputs/suwan_projectV2'
RUN=ROOT/'work/suwan_v2_reproduction'
DATA=ROOT/'work/oibc_data'
CACHE=ROOT/'work/suwan_v2_cache'
SUB=BASE/'submission'
REPEAT=ROOT/'work/suwan_v2_prediction_reproduction'

def execute(script,*args):
    subprocess.run([sys.executable,'-u',str(script),*map(str,args)],cwd=ROOT,check=True)

def main():
    assert (DATA/'sites.csv').exists(),'Extract supplied dataset into work/oibc_data first'
    os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
    if not (BASE/'pipeline.py').exists():execute(HERE/'bootstrap.py')
    execute(BASE/'run_reproduction.py','--data',DATA,'--cache',CACHE,'--run',RUN,'--submission',SUB)
    audit=BASE/'audit_compatible.py'
    if not audit.exists():audit=ROOT/'outputs/final_stage1/audit_run.py'
    execute(audit,'--data',DATA,'--run',RUN,'--submission',SUB)
    if not (REPEAT/'receipt.json').exists():
        execute(BASE/'predict.py','--data',DATA,'--run',RUN,'--cache',CACHE,'--out',REPEAT)
    execute(BASE/'verify_reproduction.py','--run',RUN,'--submission',SUB,'--reproduction',REPEAT)
    execute(HERE/'run.py','prepare')
    for stage in ['feature-fit-dev','feature-predict-dev','choose','gate']:
        execute(HERE/'run.py',stage)
    if (HERE/'ADOPTION.json').exists():
        execute(HERE/'adopt_candidate.py','--build')
    else:
        selected=json.loads((HERE/'results/frozen_before_May.json').read_text())['selected']
        if selected['variant']=='feature':
            execute(HERE/'run.py','feature-fit-production')
            execute(HERE/'run.py','feature-predict-production')
        execute(HERE/'run.py','produce')
    execute(HERE/'verify_final.py')
    execute(HERE/'report.py')

if __name__=='__main__':main()
