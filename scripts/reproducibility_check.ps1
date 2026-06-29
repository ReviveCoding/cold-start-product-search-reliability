param([string]$Config = "configs/smoke.yaml", [switch]$Keep)
$ErrorActionPreference = "Stop"
$PythonExe = if ($env:PRODUCT_SEARCH_PYTHON) { $env:PRODUCT_SEARCH_PYTHON } else { (Get-Command python).Source }
$Arguments = @("scripts/reproducibility_check.py", "--config", $Config)
if ($Keep) { $Arguments += "--keep" }
& $PythonExe @Arguments
exit $LASTEXITCODE
