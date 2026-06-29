param([string]$Config = "configs/smoke.yaml")
$ErrorActionPreference = "Stop"
python scripts/run_full_pipeline.py --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
