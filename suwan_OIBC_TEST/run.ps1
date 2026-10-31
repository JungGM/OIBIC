param([string]$Python = 'python', [string]$RunName = 'pilot_v2', [int]$MaxTrainRows = 100000)
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:OIBC_DATA_DIR = Join-Path $PSScriptRoot 'data'
& $Python (Join-Path $PSScriptRoot 'test_pipeline.py')
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
& $Python (Join-Path $PSScriptRoot 'audit_data.py')
if ($LASTEXITCODE -ne 0) { throw 'Data audit failed' }
& $Python (Join-Path $PSScriptRoot 'pipeline.py') --data $env:OIBC_DATA_DIR --max-train-rows $MaxTrainRows --out (Join-Path $PSScriptRoot "runs/$RunName")
if ($LASTEXITCODE -ne 0) { throw 'Pipeline failed' }
