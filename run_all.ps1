$ErrorActionPreference = "Stop"
$env:PYTHONPATH="d:\Summer Of Quant'26"

Write-Host "Running Stage 9..."
& "d:\Summer Of Quant'26\.venv\Scripts\python.exe" -m pipeline.stage9.pipeline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Running Stage 10..."
& "d:\Summer Of Quant'26\.venv\Scripts\python.exe" -m pipeline.stage10.pipeline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Running Stage 11..."
& "d:\Summer Of Quant'26\.venv\Scripts\python.exe" -m pipeline.stage11.pipeline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Pipeline Complete!"
