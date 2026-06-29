param(
    [ValidateSet(
        "P00_INPUTS",
        "P01_PREFLIGHT",
        "P03_VENV",
        "P04_DEPENDENCIES",
        "P05_GPU_DECISION",
        "P07_TESTS",
        "P08_SMOKE_PIPELINE",
        "P09_INTEGRATION_REPLAY",
        "P11_EVALUATION",
        "P12_INFERENCE",
        "P13_SERVICE_SMOKE",
        "P14_FINALIZE",
        "ALL"
    )]
    [string]$Stage = "ALL",

    # Keep this stable across separate invocations so markers and logs are reused.
    [string]$RunId = "main",

    # Deliberately rebuild .venv and invalidate dependent stage markers.
    [switch]$RecreateVenv,

    # Optional only. The qualified core pipeline does not need torch/GPU.
    # This switch verifies an already-installed torch CUDA runtime; it does not install torch.
    [switch]$CheckOptionalTorchGpu,

    # Re-run the requested stage and its downstream stages even if markers exist.
    [switch]$ForceRerun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
# v13: avoid unbraced variable interpolation followed by a colon in double-quoted strings.

# RunId becomes part of the output path. Keep it path-safe to prevent accidental
# traversal outside the configured output root.
if ([string]::IsNullOrWhiteSpace($RunId) -or $RunId -notmatch "^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$") {
    throw "RunId must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$ and must not contain path separators."
}
if ($PSVersionTable.PSVersion.Major -ge 7) {
    # Do not convert normal non-zero native exit codes into a PowerShell exception;
    # Invoke-NativeChecked records logs and handles exit codes consistently below.
    $PSNativeCommandUseErrorActionPreference = $false
}

# ============================================================================
# Project-local configuration
# ============================================================================

$ProjectName = "cold-start-product-search-reliability"
$RepoRoot = "C:\Users\bjw-0\Downloads\Amazon09_cold-start-product-search-reliability"
$OutputRoot = "C:\Users\bjw-0\Downloads\Project_Outputs\Amazon09_cold-start-product-search-reliability"
$VenvRoot = Join-Path $RepoRoot ".venv"
$PythonExe = Join-Path $VenvRoot "Scripts\python.exe"

$RunRoot = Join-Path $OutputRoot ".local-run\$ProjectName\$RunId"
$LogRoot = Join-Path $RunRoot "logs"
$DiagRoot = Join-Path $RunRoot "diagnostics"
$ReportRoot = Join-Path $RunRoot "reports"
$MarkerRoot = Join-Path $RunRoot "markers"
$ErrorRoot = Join-Path $RunRoot "error-bundles"
$RunLockPath = Join-Path $RunRoot "controller.lock.json"
$WheelSmokeVenv = Join-Path "C:\p9b\p14ws" $RunId
$script:RunLockHandle = $null
$script:RunLockToken = [guid]::NewGuid().ToString("N")
$script:SavedProcessEnvironment = @{}
$script:Utf8NoBom = [System.Text.UTF8Encoding]::new($false)

New-Item -ItemType Directory -Force -Path $RunRoot, $LogRoot, $DiagRoot, $ReportRoot, $MarkerRoot, $ErrorRoot | Out-Null

# Inventory only. The default validated Project 9 pipeline uses synthetic/canonical
# smoke data and must not alter these external directories.
$ExternalData = [ordered]@{
    amazon_esci = "C:\Users\bjw-0\Downloads\Project_Data\Amazon ESCI Shopping Queries Dataset_LFS"
    open_bandit = "C:\Users\bjw-0\Downloads\Project_Data\zr-obp-master_Open Bandit Dataset"
    olist = "C:\Users\bjw-0\Downloads\Project_Data\Brazilian E-Commerce Public Dataset by Olist"
    instacart = "C:\Users\bjw-0\Downloads\Project_Data\Instacart Market Basket Analysis"
}

$StageOrder = @(
    "P00_INPUTS",
    "P01_PREFLIGHT",
    "P03_VENV",
    "P04_DEPENDENCIES",
    "P05_GPU_DECISION",
    "P07_TESTS",
    "P08_SMOKE_PIPELINE",
    "P09_INTEGRATION_REPLAY",
    "P11_EVALUATION",
    "P12_INFERENCE",
    "P13_SERVICE_SMOKE",
    "P14_FINALIZE"
)

# ============================================================================
# Utility functions
# ============================================================================

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)] [string]$Path,
        [Parameter(Mandatory = $true)] [string]$Text
    )
    $Parent = Split-Path -Parent $Path
    if ($Parent) { New-Item -ItemType Directory -Force -Path $Parent | Out-Null }
    [System.IO.File]::WriteAllText($Path, $Text, $script:Utf8NoBom)
}

function Save-Json {
    param(
        [Parameter(Mandatory = $true)] [object]$Object,
        [Parameter(Mandatory = $true)] [string]$Path
    )
    $Parent = Split-Path -Parent $Path
    if ($Parent) { New-Item -ItemType Directory -Force -Path $Parent | Out-Null }
    $TempPath = "$Path.$PID.$([guid]::NewGuid().ToString('N')).tmp"
    $Json = $Object | ConvertTo-Json -Depth 30
    [System.IO.File]::WriteAllText($TempPath, $Json, $script:Utf8NoBom)
    try {
        if ([System.IO.File]::Exists($Path)) {
            [System.IO.File]::Replace($TempPath, $Path, "${Path}.bak")
        }
        else {
            [System.IO.File]::Move($TempPath, $Path)
        }
    }
    finally {
        if ([System.IO.File]::Exists($TempPath)) {
            Remove-Item -LiteralPath $TempPath -Force -ErrorAction SilentlyContinue
        }
    }
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)] [string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Initialize-ControllerEnvironment {
    # Prevent caller-level Python/pip variables from redirecting imports or installs
    # outside this repository's dedicated venv. Values are never written to disk and
    # are restored in the top-level finally block.
    $Overrides = [ordered]@{
        "PYTHONPATH" = $null
        "PYTHONHOME" = $null
        "PYTHONNOUSERSITE" = "1"
        "PYTHONUTF8" = "1"
        "PIP_TARGET" = $null
        "PIP_PREFIX" = $null
        "PIP_USER" = $null
        "PIP_REQUIRE_VIRTUALENV" = $null
        "PIP_DISABLE_PIP_VERSION_CHECK" = "1"
        "VIRTUAL_ENV" = $null
        "OMP_NUM_THREADS" = "1"
        "MKL_NUM_THREADS" = "1"
        "OPENBLAS_NUM_THREADS" = "1"
        "NUMEXPR_NUM_THREADS" = "1"
    }
    $Presence = [ordered]@{}
    foreach ($Name in $Overrides.Keys) {
        $Current = [System.Environment]::GetEnvironmentVariable($Name, "Process")
        $script:SavedProcessEnvironment[$Name] = $Current
        $Presence[$Name] = ($null -ne $Current)
        [System.Environment]::SetEnvironmentVariable($Name, $Overrides[$Name], "Process")
    }
    Save-Json -Object @{
        timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
        redacted_original_value_presence = $Presence
        applied_overrides = @($Overrides.Keys)
    } -Path (Join-Path $DiagRoot "process_environment_isolation.json")
}

function Restore-ControllerEnvironment {
    foreach ($Name in $script:SavedProcessEnvironment.Keys) {
        [System.Environment]::SetEnvironmentVariable($Name, $script:SavedProcessEnvironment[$Name], "Process")
    }
}

function Get-CoreSourceFingerprint {
    # Include all tracked source/configuration/documentation inputs used by a local run,
    # but explicitly exclude disposable environments, generated outputs, and caches.
    $ExcludedParts = @(
        ".git", ".venv", ".local-run", "artifacts", "dist", "build",
        "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".coverage"
    )
    $Rows = foreach ($Item in Get-ChildItem -LiteralPath $RepoRoot -File -Force -Recurse) {
        $Relative = $Item.FullName.Substring($RepoRoot.Length).TrimStart([char[]]@('\', '/'))
        $Parts = $Relative -split "[\\/]"
        if ($Parts | Where-Object { $_ -in $ExcludedParts -or $_ -like ".venv.*" -or $_ -like ".venv-*" -or $_ -like "*.egg-info" }) { continue }
        if ($Item.Name -like "*.pyc") { continue }
        [ordered]@{ path = $Relative.Replace("\", "/"); sha256 = Get-Sha256 -Path $Item.FullName }
    }
    $OrderedRows = @($Rows | Sort-Object path)
    $Canonical = ($OrderedRows | ConvertTo-Json -Compress -Depth 5)
    $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Canonical)
    $Hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $Digest = $Hasher.ComputeHash($Bytes)
        $Fingerprint = -join ($Digest | ForEach-Object { $_.ToString("x2") })
    }
    finally {
        $Hasher.Dispose()
    }
    return [ordered]@{ fingerprint = $Fingerprint; files = $OrderedRows }
}

function Write-StageState {
    param(
        [Parameter(Mandatory = $true)] [string]$StageId,
        [Parameter(Mandatory = $true)] [string]$Status,
        [string]$LatestSuccessfulStage = "",
        [string]$FailedCommand = "",
        [int]$ExitCode = 0,
        [string]$ResumeCommand = ""
    )
    Save-Json -Object ([ordered]@{
        project_name = $ProjectName
        run_id = $RunId
        repo_root = $RepoRoot
        output_root = $OutputRoot
        current_stage = $StageId
        stage_status = $Status
        latest_successful_stage = $LatestSuccessfulStage
        failed_command = $FailedCommand
        exit_code = $ExitCode
        resume_command = $ResumeCommand
        timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
    }) -Path (Join-Path $RunRoot "state.json")
}

function Assert-PathExists {
    param(
        [Parameter(Mandatory = $true)] [string]$Path,
        [Parameter(Mandatory = $true)] [string]$Name
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw ("Missing required {0}: {1}" -f $Name, $Path)
    }
}

function Test-StageMarker {
    param([Parameter(Mandatory = $true)] [string]$StageId)
    return (Test-Path -LiteralPath (Join-Path $MarkerRoot "$StageId.passed"))
}

function Set-StageMarker {
    param([Parameter(Mandatory = $true)] [string]$StageId)
    Write-Utf8NoBom -Path (Join-Path $MarkerRoot "$StageId.passed") -Text "PASSED $(Get-Date -Format o)"
}

function Remove-StageMarkersFrom {
    param([Parameter(Mandatory = $true)] [string]$StageId)
    $Index = [array]::IndexOf($StageOrder, $StageId)
    if ($Index -lt 0) { throw "Unknown stage for marker removal: $StageId" }
    foreach ($Name in $StageOrder[$Index..($StageOrder.Count - 1)]) {
        Remove-Item -LiteralPath (Join-Path $MarkerRoot "$Name.passed") -Force -ErrorAction SilentlyContinue
    }
}

function Test-PythonProjectImport {
    # Verify both importability and that the editable project resolves to this repository,
    # not to an identically named package from some unrelated path.
    if (-not (Test-Path -LiteralPath $PythonExe)) { return $false }
    $RepoLiteral = $RepoRoot
    $Probe = @'
from importlib import metadata
from pathlib import Path
import product_search
import xgboost
import pandas
import sklearn
import fastapi
import uvicorn

repo = Path(r"__REPO_ROOT__").resolve()
module_path = Path(product_search.__file__).resolve()
source_root = (repo / "src").resolve()
try:
    module_path.relative_to(source_root)
except ValueError as exc:
    raise SystemExit(f"product_search resolved outside this repository: {module_path} (expected under {source_root})") from exc
for distribution in ("pytest", "ruff", "build", "fastapi", "uvicorn", "xgboost-cpu"):
    metadata.version(distribution)
print(product_search.__version__)
print(module_path)
'@
    $Probe = $Probe.Replace("__REPO_ROOT__", $RepoLiteral)
    try {
        $ProjectImportProbePath = Join-Path $DiagRoot "project_import_probe.py"
        [System.IO.File]::WriteAllText($ProjectImportProbePath, $Probe, $script:Utf8NoBom)
        & $PythonExe $ProjectImportProbePath 1> (Join-Path $DiagRoot "project_import_probe.stdout.txt") 2> (Join-Path $DiagRoot "project_import_probe.stderr.txt")
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        $_ | Out-File -LiteralPath (Join-Path $DiagRoot "project_import_probe.stderr.txt") -Encoding utf8 -Append
        return $false
    }
}

function Write-ConstraintContractScript {
    # The contract script uses a JSON-encoded path literal, so spaces, Unicode, and
    # backslashes in the repository path remain valid Python syntax.
    $ConstraintProbe = Join-Path $RunRoot "validate_constraint_contract.py"
    $ConstraintsPath = Join-Path $RepoRoot "constraints\validated.txt"
    $ConstraintPathJson = ConvertTo-Json -InputObject $ConstraintsPath -Compress
    $ConstraintTemplate = @'
from importlib import metadata
from pathlib import Path
from packaging.markers import Marker

constraints = Path(__CONSTRAINTS_PATH_JSON__)
errors = []
for raw in constraints.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "==" not in line:
        continue
    requirement, separator, marker = line.partition(";")
    if separator:
        try:
            if not Marker(marker.strip()).evaluate():
                continue
        except Exception as exc:
            errors.append(f"invalid environment marker: {line} ({exc})")
            continue
    name, expected = requirement.split("==", 1)
    name = name.strip()
    expected = expected.strip()
    try:
        actual = metadata.version(name)
    except metadata.PackageNotFoundError:
        errors.append(f"missing: {name}=={expected}")
        continue
    if actual != expected:
        errors.append(f"version mismatch: {name} expected {expected}, got {actual}")
if errors:
    raise SystemExit("\n".join(errors))
print("validated constraint contract PASS")
'@
    $ConstraintContent = $ConstraintTemplate.Replace("__CONSTRAINTS_PATH_JSON__", $ConstraintPathJson)
    Write-Utf8NoBom -Path $ConstraintProbe -Text $ConstraintContent
    return $ConstraintProbe
}

function Test-ValidatedDependencyContract {
    # A P04 marker alone is not enough: users can change packages inside .venv after
    # a successful run. Verify every active validated constraint before a marker skip.
    if (-not (Test-Path -LiteralPath $PythonExe)) { return $false }
    $Probe = Write-ConstraintContractScript
    $StdOut = Join-Path $DiagRoot "constraint_contract_probe.stdout.txt"
    $StdErr = Join-Path $DiagRoot "constraint_contract_probe.stderr.txt"
    try {
        & $PythonExe $Probe 1> $StdOut 2> $StdErr
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        $_ | Out-File -LiteralPath $StdErr -Encoding utf8 -Append
        return $false
    }
}

function Begin-StageExecution {
    param([Parameter(Mandatory = $true)] [string]$StageId)
    # Any real rerun invalidates this stage's success marker and all downstream
    # evidence. This prevents a fresh pipeline from being paired with stale reports,
    # package artifacts, or finalization markers.
    Remove-StageMarkersFrom -StageId $StageId
    Write-StageState -StageId $StageId -Status "RUNNING"
}

function Repair-StaleMarkers {
    # Markers are a resume optimization, never proof that sources, dependencies, or outputs still exist.
    if (Test-Path -LiteralPath $RepoRoot) {
        $FingerprintPath = Join-Path $DiagRoot "core_source_fingerprint.json"
        $CurrentFingerprint = Get-CoreSourceFingerprint
        $SourceChanged = $false
        $ChangedPaths = @()
        if (Test-Path -LiteralPath $FingerprintPath) {
            try {
                $Previous = Get-Content -LiteralPath $FingerprintPath -Raw | ConvertFrom-Json
                $PreviousMap = @{}
                foreach ($Row in @($Previous.files)) { $PreviousMap[[string]$Row.path] = [string]$Row.sha256 }
                foreach ($Row in @($CurrentFingerprint.files)) {
                    $OldValue = if ($PreviousMap.ContainsKey([string]$Row.path)) { $PreviousMap[[string]$Row.path] } else { "MISSING" }
                    if ($OldValue -ne [string]$Row.sha256) { $ChangedPaths += [string]$Row.path }
                }
                foreach ($OldPath in $PreviousMap.Keys) {
                    if (-not (@($CurrentFingerprint.files | Where-Object { [string]$_.path -eq $OldPath }))) { $ChangedPaths += $OldPath }
                }
                $SourceChanged = ($ChangedPaths.Count -gt 0)
            }
            catch {
                $SourceChanged = $true
                $ChangedPaths = @("<unreadable previous fingerprint>")
            }
        }
        if ($SourceChanged) {
            $DependencyInputs = @("pyproject.toml", "constraints/validated.txt")
            if (@($ChangedPaths | Where-Object { $_ -in $DependencyInputs }).Count -gt 0) {
                Remove-StageMarkersFrom -StageId "P04_DEPENDENCIES"
            }
            else {
                Remove-StageMarkersFrom -StageId "P07_TESTS"
            }
            Save-Json -Object @{
                timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
                action = "invalidated_stale_markers_for_source_change"
                changed_paths = @($ChangedPaths | Sort-Object -Unique)
            } -Path (Join-Path $DiagRoot "stale_marker_repair.json")
        }
        Save-Json -Object $CurrentFingerprint -Path $FingerprintPath
    }

    if (-not (Test-Path -LiteralPath $PythonExe)) {
        Remove-StageMarkersFrom -StageId "P03_VENV"
        return
    }
    if (-not (Test-PythonProjectImport)) {
        Remove-StageMarkersFrom -StageId "P04_DEPENDENCIES"
        return
    }
    if ((Test-StageMarker -StageId "P04_DEPENDENCIES") -and -not (Test-ValidatedDependencyContract)) {
        Remove-StageMarkersFrom -StageId "P04_DEPENDENCIES"
        Save-Json -Object @{
            timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
            action = "invalidated_stale_markers_for_dependency_contract_drift"
        } -Path (Join-Path $DiagRoot "stale_marker_repair.json")
        return
    }
    if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot "artifacts\smoke\release_decision.json"))) {
        Remove-StageMarkersFrom -StageId "P08_SMOKE_PIPELINE"
        return
    }
    if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot "dist\cold_start_product_search_reliability-0.6.0-py3-none-any.whl"))) {
        Remove-StageMarkersFrom -StageId "P14_FINALIZE"
        return
    }
    if (-not (Test-Path -LiteralPath (Join-Path $ReportRoot "final_summary.json"))) {
        Remove-StageMarkersFrom -StageId "P14_FINALIZE"
        return
    }
    if (-not (Test-Path -LiteralPath (Join-Path $WheelSmokeVenv "Scripts\python.exe"))) {
        Remove-StageMarkersFrom -StageId "P14_FINALIZE"
    }
}

function Test-ShouldSkipStage {
    param([Parameter(Mandatory = $true)] [string]$StageId)
    if ($ForceRerun) { return $false }
    return (Test-StageMarker -StageId $StageId)
}

function Format-CommandForLog {
    param(
        [Parameter(Mandatory = $true)] [string]$FilePath,
        [Parameter(Mandatory = $true)] [string[]]$Arguments
    )
    $QuotedArguments = $Arguments | ForEach-Object {
        '"' + ($_ -replace '"', '\"') + '"'
    }
    return '& "' + $FilePath + '" ' + ($QuotedArguments -join ' ')
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)] [string]$StageId,
        [Parameter(Mandatory = $true)] [string]$LogName,
        [Parameter(Mandatory = $true)] [string]$FilePath,
        [Parameter(Mandatory = $true)] [string[]]$Arguments
    )

    Assert-PathExists -Path $FilePath -Name "executable for $LogName"

    $StdOut = Join-Path $LogRoot "$LogName.stdout.log"
    $StdErr = Join-Path $LogRoot "$LogName.stderr.log"
    $CommandPath = Join-Path $LogRoot "$LogName.command.txt"
    $CommandText = Format-CommandForLog -FilePath $FilePath -Arguments $Arguments
    $CommandText | Out-File -LiteralPath $CommandPath -Encoding utf8

    Write-Host ""
    Write-Host "[$StageId] RUN: $CommandText"
    Write-StageState -StageId $StageId -Status "RUNNING"

    $ExitCode = 1
    $PowerShellException = $null
    Push-Location $RepoRoot
    try {
        # Direct invocation preserves a PowerShell argument array for paths containing spaces.
        $SavedErrorActionPreference = $ErrorActionPreference
        try {
            # Native tools may emit informational stderr while still exiting 0.
            # Keep stderr in its log; decide success from $LASTEXITCODE below.
            $ErrorActionPreference = "SilentlyContinue"
            & $FilePath @Arguments 1> $StdOut 2> $StdErr
        }
        finally {
            $ErrorActionPreference = $SavedErrorActionPreference
        }
        if ($null -eq $LASTEXITCODE) {
            $ExitCode = 0
        }
        else {
            $ExitCode = [int]$LASTEXITCODE
        }
    }
    catch {
        $PowerShellException = $_
        $_ | Out-File -LiteralPath $StdErr -Encoding utf8 -Append
        $ExitCode = 1
    }
    finally {
        Pop-Location
    }

    Write-Host "[$StageId] EXIT: $ExitCode"
    Write-Host "stdout: $StdOut"
    Write-Host "stderr: $StdErr"

    if ($ExitCode -ne 0) {
        $Bundle = New-ErrorBundle -StageId $StageId -Files @($StdOut, $StdErr, $CommandPath)
        if ($PowerShellException) {
            $PowerShellException | Out-File -LiteralPath (Join-Path $Bundle "powershell_exception.txt") -Encoding utf8
        }
        Write-StageState -StageId $StageId -Status "FAILED" -FailedCommand $CommandText -ExitCode $ExitCode -ResumeCommand ".\Amazon09_run_controller_verified_v12.ps1 -Stage $StageId -RunId $RunId"
        throw "Stage $StageId failed with exit code $ExitCode. Error bundle: $Bundle"
    }
}

function Get-PythonLauncher {
    foreach ($RequestedVersion in @("-3.13", "-3.12", "-3.11")) {
        try {
            $VersionOutput = & py $RequestedVersion --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                return @{ executable = "py"; arguments = @($RequestedVersion); version = ($VersionOutput | Out-String).Trim() }
            }
        }
        catch { }
    }
    try {
        $VersionOutput = & python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            return @{ executable = "python"; arguments = @(); version = ($VersionOutput | Out-String).Trim() }
        }
    }
    catch { }
    throw "Python 3.11, 3.12, or 3.13 was not found. Install a supported CPython version and rerun P03_VENV."
}

function Assert-SupportedPython {
    param([Parameter(Mandatory = $true)] [string]$PythonPath)
    $ProbePath = Join-Path $DiagRoot "selected_python_version_probe.py"
    $ProbeLines = @(
        'import sys',
        'print(".".join(map(str, sys.version_info[:3])))',
        'raise SystemExit(0 if ((3, 11) <= sys.version_info[:2] < (3, 14)) else 2)'
    )
    $Probe = $ProbeLines -join [Environment]::NewLine
    [System.IO.File]::WriteAllText($ProbePath, $Probe, $script:Utf8NoBom)
    & $PythonPath $ProbePath 1> (Join-Path $DiagRoot "selected_python_version.txt") 2> (Join-Path $DiagRoot "selected_python_version.stderr.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "Project requires Python >=3.11,<3.14. See $DiagRoot\selected_python_version.stderr.txt"
    }
}

function Stop-ProjectUvicornResidue {
    # Restrict cleanup to this controller's own .venv or repo path. Never kill an unrelated
    # manually launched Uvicorn service merely because it has the same import target.
    $Killed = @()
    $PythonPattern = [regex]::Escape($PythonExe)
    $RepoPattern = [regex]::Escape($RepoRoot)
    try {
        Get-CimInstance Win32_Process | Where-Object {
            $_.CommandLine -and
            $_.CommandLine -match "product_search\.serving\.app:app" -and
            ($_.CommandLine -match $PythonPattern -or $_.CommandLine -match $RepoPattern)
        } | ForEach-Object {
            try {
                Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop
                $Killed += $_.ProcessId
            }
            catch { }
        }
    }
    catch { }
    Save-Json -Object @{ timestamp_utc = (Get-Date).ToUniversalTime().ToString("o"); killed_process_ids = $Killed } -Path (Join-Path $DiagRoot "uvicorn_residue_cleanup.json")
}

function Acquire-RunLock {
    # FileMode.CreateNew plus FileShare.None guarantees one live controller per RunId.
    # If a previous controller died after creating a malformed/empty file, distinguish
    # it from a currently held handle by attempting a separate exclusive open before
    # reclaiming it.
    try {
        $Payload = @{ run_id = $RunId; process_id = $PID; token = $script:RunLockToken; started_at_utc = (Get-Date).ToUniversalTime().ToString("o") } | ConvertTo-Json -Compress
        $script:RunLockHandle = [System.IO.File]::Open($RunLockPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
        $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Payload)
        $script:RunLockHandle.Write($Bytes, 0, $Bytes.Length)
        $script:RunLockHandle.Flush()
    }
    catch [System.IO.IOException] {
        $ProbeHandle = $null
        $CanReclaim = $false
        try {
            # Succeeds only when the existing path is not actively held with an
            # exclusive sharing mode. This makes malformed stale files recoverable.
            $ProbeHandle = [System.IO.File]::Open($RunLockPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
            $CanReclaim = $true
        }
        catch {
            $CanReclaim = $false
        }
        finally {
            if ($ProbeHandle) { $ProbeHandle.Dispose() }
        }

        if ($CanReclaim) {
            $StalePath = "$RunLockPath.stale-" + (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + [guid]::NewGuid().ToString("N")
            Move-Item -LiteralPath $RunLockPath -Destination $StalePath -Force
            Acquire-RunLock
            return
        }
        throw "Another controller currently owns RunId=$RunId. Lock: $RunLockPath. Do not run two controllers against the same run directory."
    }
}

function Release-RunLock {
    if ($script:RunLockHandle) {
        $script:RunLockHandle.Dispose()
        $script:RunLockHandle = $null
    }
    if (Test-Path -LiteralPath $RunLockPath) {
        # Do not delete a different controller's lock if a race occurred after
        # releasing this process's handle.
        $OwnedByThisController = $false
        try {
            $Existing = Get-Content -LiteralPath $RunLockPath -Raw | ConvertFrom-Json
            $OwnedByThisController = ($Existing.token -eq $script:RunLockToken)
        }
        catch { $OwnedByThisController = $false }
        if ($OwnedByThisController) {
            Remove-Item -LiteralPath $RunLockPath -Force -ErrorAction SilentlyContinue
        }
    }
}

function New-ErrorBundle {
    param(
        [Parameter(Mandatory = $true)] [string]$StageId,
        [Parameter(Mandatory = $true)] [string[]]$Files
    )
    $Bundle = Join-Path $ErrorRoot ("$StageId-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
    New-Item -ItemType Directory -Force -Path $Bundle | Out-Null
    foreach ($File in $Files) {
        if (Test-Path -LiteralPath $File) {
            Copy-Item -LiteralPath $File -Destination $Bundle -Force -ErrorAction SilentlyContinue
        }
    }
    foreach ($Diagnostic in @(
        (Join-Path $RunRoot "state.json"),
        (Join-Path $DiagRoot "core_source_fingerprint.json"),
        (Join-Path $DiagRoot "python_inventory.txt"),
        (Join-Path $DiagRoot "gpu_info.txt"),
        (Join-Path $ReportRoot "pip_install_report.json")
    )) {
        if (Test-Path -LiteralPath $Diagnostic) {
            Copy-Item -LiteralPath $Diagnostic -Destination $Bundle -Force -ErrorAction SilentlyContinue
        }
    }
    return $Bundle
}

function Complete-Stage {
    param(
        [Parameter(Mandatory = $true)] [string]$StageId,
        [Parameter(Mandatory = $true)] [string]$NextStage
    )
    Write-StageState -StageId $StageId -Status "PASSED" -LatestSuccessfulStage $StageId -ResumeCommand ".\Amazon09_run_controller_verified_v12.ps1 -Stage $NextStage -RunId $RunId"
    Set-StageMarker $StageId
}

# ============================================================================
# Stage functions
# ============================================================================

function Run-P00-Inputs {
    $StageId = "P00_INPUTS"
    if (Test-ShouldSkipStage -StageId $StageId) { Write-Host "$StageId already passed for RunId=$RunId."; return }
    Begin-StageExecution -StageId $StageId

    Assert-PathExists -Path $RepoRoot -Name "repository root"
    foreach ($RelativePath in @(
        "pyproject.toml",
        "constraints\validated.txt",
        "configs\smoke.yaml",
        "scripts\run_tests.py",
        "scripts\run_full_pipeline.py",
        "scripts\integration_validation.py",
        "scripts\reproducibility_check.py",
        "scripts\run_policy_sensitivity.py",
        "scripts\uvicorn_validation.py",
        "scripts\build_release.py",
        "scripts\verify_build_reproducibility.py",
        "scripts\validate_candidate_handoff.py",
        "release_candidate_handoff.json"
    )) {
        Assert-PathExists -Path (Join-Path $RepoRoot $RelativePath) -Name $RelativePath
    }

    $Inventory = foreach ($Key in $ExternalData.Keys) {
        [pscustomobject]@{
            name = $Key
            path = $ExternalData[$Key]
            exists = Test-Path -LiteralPath $ExternalData[$Key]
            use_in_default_pipeline = $false
            note = "Inventory only; default Project 9 smoke pipeline uses synthetic/canonical data."
        }
    }
    Save-Json -Object $Inventory -Path (Join-Path $DiagRoot "external_dataset_inventory.json")

    $Fingerprint = Get-CoreSourceFingerprint
    Save-Json -Object $Fingerprint -Path (Join-Path $DiagRoot "core_source_fingerprint.json")

    Complete-Stage -StageId $StageId -NextStage "P01_PREFLIGHT"
}

function Run-P01-Preflight {
    $StageId = "P01_PREFLIGHT"
    if (Test-ShouldSkipStage -StageId $StageId) { Write-Host "$StageId already passed for RunId=$RunId."; return }
    Begin-StageExecution -StageId $StageId

    $PythonInventory = Join-Path $DiagRoot "python_inventory.txt"
    "=== py launcher ===" | Out-File -LiteralPath $PythonInventory -Encoding utf8
    try { & py "-0p" 2>&1 | Out-File -LiteralPath $PythonInventory -Encoding utf8 -Append } catch { $_ | Out-File -LiteralPath $PythonInventory -Encoding utf8 -Append }
    "=== python ===" | Out-File -LiteralPath $PythonInventory -Encoding utf8 -Append
    try { python --version 2>&1 | Out-File -LiteralPath $PythonInventory -Encoding utf8 -Append } catch { $_ | Out-File -LiteralPath $PythonInventory -Encoding utf8 -Append }

    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        nvidia-smi 2>&1 | Out-File -LiteralPath (Join-Path $DiagRoot "gpu_info.txt") -Encoding utf8
    }
    else {
        "nvidia-smi not found on PATH." | Out-File -LiteralPath (Join-Path $DiagRoot "gpu_info.txt") -Encoding utf8
    }

    Get-PSDrive -PSProvider FileSystem | Select-Object -Property @("Name", "Root", "Used", "Free") | ConvertTo-Json -Depth 5 |
        Out-File -LiteralPath (Join-Path $DiagRoot "disk_free.json") -Encoding utf8
    Copy-Item -LiteralPath (Join-Path $RepoRoot "constraints\validated.txt") -Destination (Join-Path $DiagRoot "constraints_validated.txt") -Force
    Copy-Item -LiteralPath (Join-Path $RepoRoot "pyproject.toml") -Destination (Join-Path $DiagRoot "pyproject.toml") -Force

    Complete-Stage -StageId $StageId -NextStage "P03_VENV"
}

function Run-P03-Venv {
    $StageId = "P03_VENV"
    if ((Test-ShouldSkipStage -StageId $StageId) -and -not $RecreateVenv) { Write-Host "$StageId already passed for RunId=$RunId."; return }
    Begin-StageExecution -StageId $StageId


    if ((Test-Path -LiteralPath $PythonExe) -and (-not $RecreateVenv)) {
        Write-Host "Reusing existing virtual environment: $VenvRoot"
        Assert-SupportedPython -PythonPath $PythonExe
        Complete-Stage -StageId $StageId -NextStage "P04_DEPENDENCIES"
        return
    }

    if (Test-Path -LiteralPath $VenvRoot) {
        # A directory can exist without a usable Scripts\python.exe after a cancelled install.
        # Never attempt to build into that ambiguous state; preserve it for inspection instead.
        $BackupReason = if ($RecreateVenv) { "recreated" } else { "incomplete" }
        $Backup = "$VenvRoot.$BackupReason-" + (Get-Date -Format "yyyyMMdd-HHmmss")
        Move-Item -LiteralPath $VenvRoot -Destination $Backup
        Write-Host "Moved existing $BackupReason virtual environment to: $Backup"
    }

    $Launcher = Get-PythonLauncher
    Save-Json -Object $Launcher -Path (Join-Path $DiagRoot "selected_python_launcher.json")
    if ($Launcher.executable -eq "py") {
        & py @($Launcher.arguments) -m venv $VenvRoot
    }
    else {
        & python -m venv $VenvRoot
    }
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed with exit code $LASTEXITCODE." }

    Assert-PathExists -Path $PythonExe -Name "venv Python executable"
    Assert-SupportedPython -PythonPath $PythonExe
    Remove-StageMarkersFrom -StageId "P04_DEPENDENCIES"
    Complete-Stage -StageId $StageId -NextStage "P04_DEPENDENCIES"
}

function Run-P04-Dependencies {
    $StageId = "P04_DEPENDENCIES"
    if ((Test-ShouldSkipStage -StageId $StageId) -and (Test-PythonProjectImport) -and (Test-ValidatedDependencyContract)) { Write-Host "$StageId already passed for RunId=$RunId."; return }
    Begin-StageExecution -StageId $StageId
    Assert-PathExists -Path $PythonExe -Name "venv Python executable"


    Invoke-NativeChecked -StageId $StageId -LogName "ensurepip" -FilePath $PythonExe -Arguments @("-m", "ensurepip", "--upgrade")
    Invoke-NativeChecked -StageId $StageId -LogName "pip_upgrade" -FilePath $PythonExe -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    Invoke-NativeChecked -StageId $StageId -LogName "pip_install_project" -FilePath $PythonExe -Arguments @(
        "-m", "pip", "install", "--report", (Join-Path $ReportRoot "pip_install_report.json"),
        "-c", (Join-Path $RepoRoot "constraints\validated.txt"), "-e", ".[dev]"
    )
    Invoke-NativeChecked -StageId $StageId -LogName "pip_check" -FilePath $PythonExe -Arguments @("-m", "pip", "check")
    Invoke-NativeChecked -StageId $StageId -LogName "pip_freeze" -FilePath $PythonExe -Arguments @("-m", "pip", "freeze", "--all")
    Invoke-NativeChecked -StageId $StageId -LogName "pip_inspect" -FilePath $PythonExe -Arguments @("-m", "pip", "inspect")

    $ImportProbe = Join-Path $RunRoot "import_probe.py"
    $ImportProbeContent = @'
import importlib.metadata as md
import product_search
print("product_search", product_search.__version__)
for package in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost-cpu", "fastapi", "uvicorn", "pydantic", "pytest", "ruff", "build"]:
    print(package, md.version(package))
'@
    Write-Utf8NoBom -Path $ImportProbe -Text $ImportProbeContent
    Invoke-NativeChecked -StageId $StageId -LogName "import_probe" -FilePath $PythonExe -Arguments @($ImportProbe)
    if (-not (Test-PythonProjectImport)) {
        throw "Editable project import origin validation failed. See $DiagRoot\project_import_probe.stderr.txt"
    }

    $ConstraintProbe = Write-ConstraintContractScript
    Invoke-NativeChecked -StageId $StageId -LogName "constraint_contract" -FilePath $PythonExe -Arguments @($ConstraintProbe)

    Remove-StageMarkersFrom -StageId "P05_GPU_DECISION"
    Complete-Stage -StageId $StageId -NextStage "P05_GPU_DECISION"
}

function Run-P05-GpuDecision {
    $StageId = "P05_GPU_DECISION"
    if (Test-ShouldSkipStage -StageId $StageId) { Write-Host "$StageId already passed for RunId=$RunId."; return }
    Begin-StageExecution -StageId $StageId

    if (-not $CheckOptionalTorchGpu) {
        Save-Json -Object @{
            status = "CPU_CORE_PATH"
            reason = "The default Project 9 pipeline uses retrieval, XGBoost, calibration, simulation, OPE, and FastAPI. Torch/GPU is not required for its qualified core path."
            nvidia_smi_present = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
            optional_gpu_note = "Run P05_GPU_DECISION with -CheckOptionalTorchGpu only after installing a compatible torch CUDA wheel from the official PyTorch selector."
        } -Path (Join-Path $ReportRoot "gpu_decision.json")
        Complete-Stage -StageId $StageId -NextStage "P07_TESTS"
        return
    }

    $TorchProbe = Join-Path $RunRoot "torch_probe.py"
    $TorchProbeContent = @'
try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "Optional torch GPU check requested, but torch is not installed. "
        "Use the current official PyTorch selector for your Windows/Python/CUDA combination. " + str(exc)
    )
print("torch", torch.__version__)
print("cuda_runtime", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("torch is installed but CUDA is unavailable")
print("device_name", torch.cuda.get_device_name(0))
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
a = torch.randn((1024, 1024), device="cuda", requires_grad=True)
b = torch.randn((1024, 1024), device="cuda", requires_grad=True)
loss = (a @ b).float().square().mean()
loss.backward()
torch.cuda.synchronize()
print("peak_memory_bytes", torch.cuda.max_memory_allocated())
'@
    Write-Utf8NoBom -Path $TorchProbe -Text $TorchProbeContent
    Invoke-NativeChecked -StageId $StageId -LogName "optional_torch_gpu_probe" -FilePath $PythonExe -Arguments @($TorchProbe)
    Complete-Stage -StageId $StageId -NextStage "P07_TESTS"
}

function Run-P07-Tests {
    $StageId = "P07_TESTS"
    if (Test-ShouldSkipStage -StageId $StageId) { Write-Host "$StageId already passed for RunId=$RunId."; return }
    Begin-StageExecution -StageId $StageId
    Invoke-NativeChecked -StageId $StageId -LogName "ruff" -FilePath $PythonExe -Arguments @("-m", "ruff", "check", "src", "tests", "scripts")
    Invoke-NativeChecked -StageId $StageId -LogName "compileall" -FilePath $PythonExe -Arguments @("-m", "compileall", "-q", "src", "scripts", "tests")
    Invoke-NativeChecked -StageId $StageId -LogName "tests" -FilePath $PythonExe -Arguments @("scripts\run_tests.py")
    Complete-Stage -StageId $StageId -NextStage "P08_SMOKE_PIPELINE"
}

function Run-P08-SmokePipeline {
    $StageId = "P08_SMOKE_PIPELINE"
    if (Test-ShouldSkipStage -StageId $StageId) { Write-Host "$StageId already passed for RunId=$RunId."; return }
    Begin-StageExecution -StageId $StageId
    Invoke-NativeChecked -StageId $StageId -LogName "pipeline" -FilePath $PythonExe -Arguments @("scripts\run_full_pipeline.py", "--config", "configs\smoke.yaml")
    Invoke-NativeChecked -StageId $StageId -LogName "handoff_preflight" -FilePath $PythonExe -Arguments @("scripts\validate_candidate_handoff.py", "--handoff", "release_candidate_handoff.json", "--allow-missing-build-artifacts")
    Complete-Stage -StageId $StageId -NextStage "P09_INTEGRATION_REPLAY"
}

function Run-P09-IntegrationReplay {
    $StageId = "P09_INTEGRATION_REPLAY"
    if (Test-ShouldSkipStage -StageId $StageId) { Write-Host "$StageId already passed for RunId=$RunId."; return }
    Begin-StageExecution -StageId $StageId
    try {
        Invoke-NativeChecked -StageId $StageId -LogName "integration_validation" -FilePath $PythonExe -Arguments @("scripts\integration_validation.py", "--config", "configs\smoke.yaml", "--skip-pipeline")
    }
    finally {
        Stop-ProjectUvicornResidue
    }
    Invoke-NativeChecked -StageId $StageId -LogName "reproducibility" -FilePath $PythonExe -Arguments @("scripts\reproducibility_check.py", "--config", "configs\smoke.yaml")
    Complete-Stage -StageId $StageId -NextStage "P11_EVALUATION"
}

function Run-P11-Evaluation {
    $StageId = "P11_EVALUATION"
    if (Test-ShouldSkipStage -StageId $StageId) { Write-Host "$StageId already passed for RunId=$RunId."; return }
    Begin-StageExecution -StageId $StageId
    Invoke-NativeChecked -StageId $StageId -LogName "policy_sensitivity" -FilePath $PythonExe -Arguments @("scripts\run_policy_sensitivity.py", "--config", "configs\smoke.yaml")
    foreach ($RelativePath in @(
        "artifacts\smoke\release_decision.json",
        "artifacts\smoke\metrics.json",
        "artifacts\smoke\dynamic_summary.json",
        "artifacts\smoke\ope_metrics.json",
        "artifacts\smoke\serving_benchmark.json",
        "artifacts\smoke\manifest.json"
    )) {
        $Source = Join-Path $RepoRoot $RelativePath
        if (Test-Path -LiteralPath $Source) {
            Copy-Item -LiteralPath $Source -Destination (Join-Path $ReportRoot ($RelativePath -replace "\\", "_")) -Force
        }
    }
    Complete-Stage -StageId $StageId -NextStage "P12_INFERENCE"
}

function Run-P12-Inference {
    $StageId = "P12_INFERENCE"
    if (Test-ShouldSkipStage -StageId $StageId) { Write-Host "$StageId already passed for RunId=$RunId."; return }
    Begin-StageExecution -StageId $StageId
    $Script = Join-Path $RunRoot "inference_smoke.py"
    $InferenceContent = @'
from pathlib import Path
from product_search.serving.app import SearchService
service = SearchService(Path("artifacts/smoke"))
response = service.search("wireless headphones", 5)
print({
    "model_version": service.model_version,
    "results": len(response.results),
    "fallback_used": response.fallback_used,
    "top_product_id": response.results[0].product_id if response.results else None,
})
if response.fallback_used:
    raise SystemExit("unexpected fallback")
if len(response.results) != 5:
    raise SystemExit("expected exactly five results")
'@
    Write-Utf8NoBom -Path $Script -Text $InferenceContent
    $env:PRODUCT_SEARCH_STRICT_ENV = "1"
    $env:PRODUCT_SEARCH_VERIFY_ARTIFACTS = "1"
    Invoke-NativeChecked -StageId $StageId -LogName "inference_smoke" -FilePath $PythonExe -Arguments @($Script)
    Complete-Stage -StageId $StageId -NextStage "P13_SERVICE_SMOKE"
}

function Run-P13-ServiceSmoke {
    $StageId = "P13_SERVICE_SMOKE"
    if (Test-ShouldSkipStage -StageId $StageId) { Write-Host "$StageId already passed for RunId=$RunId."; return }
    Begin-StageExecution -StageId $StageId
    try {
        Invoke-NativeChecked -StageId $StageId -LogName "uvicorn_single_worker" -FilePath $PythonExe -Arguments @(
            "scripts\uvicorn_validation.py",
            "--artifact-dir", "artifacts\smoke",
            "--requests", "16",
            "--concurrency", "4",
            "--workers", "1"
        )
    }
    finally {
        Stop-ProjectUvicornResidue
    }
    Complete-Stage -StageId $StageId -NextStage "P14_FINALIZE"
}

function Run-P14-Finalize {
    $StageId = "P14_FINALIZE"
    if (Test-ShouldSkipStage -StageId $StageId) { Write-Host "$StageId already passed for RunId=$RunId."; return }
    Begin-StageExecution -StageId $StageId
    Invoke-NativeChecked -StageId $StageId -LogName "build_reproducibility" -FilePath $PythonExe -Arguments @("scripts\verify_build_reproducibility.py")
    Invoke-NativeChecked -StageId $StageId -LogName "build_release" -FilePath $PythonExe -Arguments @("scripts\build_release.py", "--output-dir", "dist")
    Invoke-NativeChecked -StageId $StageId -LogName "handoff_strict" -FilePath $PythonExe -Arguments @("scripts\validate_candidate_handoff.py", "--handoff", "release_candidate_handoff.json")

    $ExpectedDist = @(
        (Join-Path $RepoRoot "dist\cold_start_product_search_reliability-0.6.0-py3-none-any.whl"),
        (Join-Path $RepoRoot "dist\cold_start_product_search_reliability-0.6.0.tar.gz")
    )
    foreach ($Path in $ExpectedDist) { Assert-PathExists -Path $Path -Name "distribution artifact" }

    # A wheel that only builds is not sufficient. Install it into an independent disposable
    # venv under the configured local-run directory, then strict-load the smoke artifacts.
    if (Test-Path -LiteralPath $WheelSmokeVenv) {
        Remove-Item -LiteralPath $WheelSmokeVenv -Force -Recurse
    }
    Invoke-NativeChecked -StageId $StageId -LogName "wheel_smoke_venv" -FilePath $PythonExe -Arguments @("-m", "venv", $WheelSmokeVenv)
    $WheelPython = Join-Path $WheelSmokeVenv "Scripts\python.exe"
    Assert-PathExists -Path $WheelPython -Name "wheel smoke Python executable"
    Invoke-NativeChecked -StageId $StageId -LogName "wheel_smoke_ensurepip" -FilePath $WheelPython -Arguments @("-m", "ensurepip", "--upgrade")
    Invoke-NativeChecked -StageId $StageId -LogName "wheel_smoke_pip_upgrade" -FilePath $WheelPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    Invoke-NativeChecked -StageId $StageId -LogName "wheel_smoke_install" -FilePath $WheelPython -Arguments @(
        "-m", "pip", "install",
        "-c", (Join-Path $RepoRoot "constraints\validated.txt"),
        $ExpectedDist[0]
    )
    Invoke-NativeChecked -StageId $StageId -LogName "wheel_smoke_pip_check" -FilePath $WheelPython -Arguments @("-m", "pip", "check")

    $WheelSmokeScript = Join-Path $RunRoot "wheel_strict_smoke.py"
    $ArtifactLiteral = Join-Path $RepoRoot "artifacts\smoke"
    $WheelSmokeContent = @'
from pathlib import Path
import product_search
from product_search.serving.app import SearchService

repo = Path(r"__REPO_ROOT__").resolve()
artifact_dir = Path(r"__ARTIFACT_DIR__").resolve()
package_path = Path(product_search.__file__).resolve()
try:
    package_path.relative_to(repo)
except ValueError:
    pass
else:
    raise SystemExit(f"wheel smoke unexpectedly imported source-tree package: {package_path}")
service = SearchService(artifact_dir)
response = service.search("wireless headphones", 5)
print({
    "package_path": str(package_path),
    "model_version": service.model_version,
    "results": len(response.results),
    "fallback_used": response.fallback_used,
})
if response.fallback_used:
    raise SystemExit("wheel smoke unexpectedly used fallback")
if len(response.results) != 5:
    raise SystemExit("wheel smoke expected exactly five results")
'@
    $WheelSmokeContent = $WheelSmokeContent.Replace("__REPO_ROOT__", $RepoRoot).Replace("__ARTIFACT_DIR__", $ArtifactLiteral)
    Write-Utf8NoBom -Path $WheelSmokeScript -Text $WheelSmokeContent
    $env:PRODUCT_SEARCH_STRICT_ENV = "1"
    $env:PRODUCT_SEARCH_VERIFY_ARTIFACTS = "1"
    Invoke-NativeChecked -StageId $StageId -LogName "wheel_strict_smoke" -FilePath $WheelPython -Arguments @($WheelSmokeScript)

    Save-Json -Object @{
        final_status = "LOCAL_PIPELINE_COMPLETED"
        run_id = $RunId
        repo_root = $RepoRoot
        reports = $ReportRoot
        logs = $LogRoot
        generated_wheel = $ExpectedDist[0]
        generated_sdist = $ExpectedDist[1]
        wheel_smoke_venv = $WheelSmokeVenv
        known_external_gates = @(
            "GitHub-hosted workflow execution",
            "Docker build",
            "native Windows CI runner parity",
            "optional GPU/Qwen/FAISS path"
        )
        completed_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    } -Path (Join-Path $ReportRoot "final_summary.json")

    Complete-Stage -StageId $StageId -NextStage "COMPLETED"
    Write-Host ""
    Write-Host "ALL LOCAL CORE STAGES COMPLETED"
    Write-Host "Reports: $ReportRoot"
    Write-Host "Logs:    $LogRoot"
}

# ============================================================================
# Pre-dispatch stale-state repair and dispatcher
# ============================================================================

Acquire-RunLock
try {
    Initialize-ControllerEnvironment
    if ($RecreateVenv) {
        Remove-StageMarkersFrom -StageId "P03_VENV"
    }
    Repair-StaleMarkers

    Write-Host "Project: $ProjectName"
    Write-Host "Repo:    $RepoRoot"
    Write-Host "RunId:   $RunId"
    Write-Host "Stage:   $Stage"

    switch ($Stage) {
        "P00_INPUTS" { Run-P00-Inputs }
        "P01_PREFLIGHT" { Run-P00-Inputs; Run-P01-Preflight }
        "P03_VENV" { Run-P00-Inputs; Run-P01-Preflight; Run-P03-Venv }
        "P04_DEPENDENCIES" { Run-P00-Inputs; Run-P01-Preflight; Run-P03-Venv; Run-P04-Dependencies }
        "P05_GPU_DECISION" { Run-P00-Inputs; Run-P01-Preflight; Run-P03-Venv; Run-P04-Dependencies; Run-P05-GpuDecision }
        "P07_TESTS" { Run-P00-Inputs; Run-P01-Preflight; Run-P03-Venv; Run-P04-Dependencies; Run-P05-GpuDecision; Run-P07-Tests }
        "P08_SMOKE_PIPELINE" { Run-P00-Inputs; Run-P01-Preflight; Run-P03-Venv; Run-P04-Dependencies; Run-P05-GpuDecision; Run-P07-Tests; Run-P08-SmokePipeline }
        "P09_INTEGRATION_REPLAY" { Run-P00-Inputs; Run-P01-Preflight; Run-P03-Venv; Run-P04-Dependencies; Run-P05-GpuDecision; Run-P07-Tests; Run-P08-SmokePipeline; Run-P09-IntegrationReplay }
        "P11_EVALUATION" { Run-P00-Inputs; Run-P01-Preflight; Run-P03-Venv; Run-P04-Dependencies; Run-P05-GpuDecision; Run-P07-Tests; Run-P08-SmokePipeline; Run-P09-IntegrationReplay; Run-P11-Evaluation }
        "P12_INFERENCE" { Run-P00-Inputs; Run-P01-Preflight; Run-P03-Venv; Run-P04-Dependencies; Run-P05-GpuDecision; Run-P07-Tests; Run-P08-SmokePipeline; Run-P09-IntegrationReplay; Run-P11-Evaluation; Run-P12-Inference }
        "P13_SERVICE_SMOKE" { Run-P00-Inputs; Run-P01-Preflight; Run-P03-Venv; Run-P04-Dependencies; Run-P05-GpuDecision; Run-P07-Tests; Run-P08-SmokePipeline; Run-P09-IntegrationReplay; Run-P11-Evaluation; Run-P12-Inference; Run-P13-ServiceSmoke }
        "P14_FINALIZE" { Run-P00-Inputs; Run-P01-Preflight; Run-P03-Venv; Run-P04-Dependencies; Run-P05-GpuDecision; Run-P07-Tests; Run-P08-SmokePipeline; Run-P09-IntegrationReplay; Run-P11-Evaluation; Run-P12-Inference; Run-P13-ServiceSmoke; Run-P14-Finalize }
        "ALL" { Run-P00-Inputs; Run-P01-Preflight; Run-P03-Venv; Run-P04-Dependencies; Run-P05-GpuDecision; Run-P07-Tests; Run-P08-SmokePipeline; Run-P09-IntegrationReplay; Run-P11-Evaluation; Run-P12-Inference; Run-P13-ServiceSmoke; Run-P14-Finalize }
    }
}
finally {
    try {
        Release-RunLock
    }
    finally {
        Restore-ControllerEnvironment
    }
}
