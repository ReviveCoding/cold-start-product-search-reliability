param(
    [string]$ControllerPath = "C:\Users\bjw-0\Downloads\Amazon09_cold-start-product-search-reliability\Amazon09_run_controller_verified_v13.ps1",
    [string]$RepoRoot = "C:\Users\bjw-0\Downloads\Amazon09_cold-start-product-search-reliability"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-ExistingPath {
    param(
        [Parameter(Mandatory = $true)] [string]$Path,
        [Parameter(Mandatory = $true)] [string]$Name
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw ("{0} not found: {1}" -f $Name, $Path)
    }
}

function Get-ParseResult {
    param([Parameter(Mandatory = $true)] [string]$Path)

    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile(
        $Path,
        [ref]$tokens,
        [ref]$errors
    ) | Out-Null

    return [pscustomobject]@{
        tokens = @($tokens)
        errors = @($errors)
    }
}

function Assert-ParseSuccess {
    param(
        [Parameter(Mandatory = $true)] [pscustomobject]$ParseResult,
        [Parameter(Mandatory = $true)] [string]$Label
    )

    if ($ParseResult.errors.Count -gt 0) {
        $ParseResult.errors | Format-List *
        throw ("PowerShell parser found {0} error(s) in {1}. Do not run it." -f $ParseResult.errors.Count, $Label)
    }
}

Assert-ExistingPath -Path $ControllerPath -Name "Controller"
Assert-ExistingPath -Path $RepoRoot -Name "Repository root"

# Parser-based validation is deliberate: $script:<name> and $env:<name> are legal
# scoped-variable syntax. A raw regular expression cannot distinguish those from a
# malformed interpolated variable such as "$Name:" reliably. ParseFile reports the
# latter as an InvalidVariableReferenceWithDrive parse error.
$ControllerParse = Get-ParseResult -Path $ControllerPath
Assert-ParseSuccess -ParseResult $ControllerParse -Label "the controller"

$SelfParse = Get-ParseResult -Path $PSCommandPath
Assert-ParseSuccess -ParseResult $SelfParse -Label "this preflight script"

$Required = @(
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
)

$Missing = @(
    $Required | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $RepoRoot $_))
    }
)
if ($Missing.Count -gt 0) {
    throw ("Repository is incomplete. Missing: {0}" -f ($Missing -join ', '))
}

$Text = Get-Content -LiteralPath $ControllerPath -Raw
$StaticChecks = [ordered]@{
    parser_reports_no_errors = ($ControllerParse.errors.Count -eq 0)
    parser_based_variable_colon_validation = $true
    stable_run_id = ($Text -match '\[string\]\$RunId\s*=\s*"main"')
    run_id_is_path_safe = ($Text -match 'RunId must match')
    no_start_process_argument_reconstruction = -not ($Text -match 'Start-Process')
    uses_direct_native_argument_arrays = ($Text -match '& \$FilePath @Arguments')
    uses_atomic_json_replace = ($Text -match '\[System\.IO\.File\]::Replace')
    uses_no_bom_json_writer = ($Text -match 'UTF8Encoding\]::new\(\$false\)')
    includes_run_lock = ($Text -match 'Acquire-RunLock') -and ($Text -match 'Release-RunLock')
    restricts_uvicorn_residue_cleanup = ($Text -match 'PythonPattern') -and ($Text -match 'RepoPattern')
    validates_editable_import_origin = ($Text -match 'resolved outside this repository')
    fingerprints_full_source_tree = ($Text -match 'Get-ChildItem -LiteralPath \$RepoRoot -File -Force -Recurse')
    excludes_backup_venvs_from_fingerprint = ($Text -match '\.venv\.\*')
    validates_source_only_handoff = ($Text -match '--allow-missing-build-artifacts')
    validates_strict_handoff_after_build = ($Text -match 'LogName "handoff_strict"')
    validates_constraint_versions = ($Text -match 'validated constraint contract PASS')
    builds_and_clean_installs_wheel = ($Text -match 'wheel_smoke_install') -and ($Text -match 'wheel_smoke_pip_check')
    wheel_import_origin_guard = ($Text -match 'wheel smoke unexpectedly imported source-tree package')
    uses_ensurepip_before_pip = ($Text -match 'ensurepip')
    uses_single_separator_trimstart = ($Text -match 'TrimStart\(\[char\[\]\]@\(''\\'', ''/''\)\)')
    does_not_import_ruff_as_python_module = -not ($Text -match 'import\s+ruff')
    optional_gpu_is_opt_in = ($Text -match '\[switch\]\$CheckOptionalTorchGpu')
    invalidates_downstream_markers_on_rerun = ($Text -match 'function Begin-StageExecution') -and ($Text -match 'Remove-StageMarkersFrom -StageId \$StageId')
    every_execution_stage_begins_with_invalidation = ([regex]::Matches($Text, 'function Run-P(?:00-Inputs|01-Preflight|03-Venv|04-Dependencies|05-GpuDecision|07-Tests|08-SmokePipeline|09-IntegrationReplay|11-Evaluation|12-Inference|13-ServiceSmoke|14-Finalize) \{[\s\S]{0,500}?Begin-StageExecution -StageId \$StageId').Count -eq 12)
    validates_dependency_contract_before_marker_skip = ($Text -match 'function Test-ValidatedDependencyContract') -and ($Text -match 'Test-ValidatedDependencyContract\)')
    uses_json_encoded_constraint_path = ($Text -match 'ConvertTo-Json -InputObject \$ConstraintsPath -Compress')
    recovers_malformed_stale_lock = ($Text -match 'CanReclaim') -and ($Text -match 'FileMode\]::Open') -and ($Text -match 'stale-')
    release_only_removes_owned_lock = ($Text -match 'OwnedByThisController') -and ($Text -match 'RunLockToken')
    isolates_caller_python_and_pip_environment = ($Text -match 'function Initialize-ControllerEnvironment') -and ($Text -match 'PYTHONPATH') -and ($Text -match 'PIP_TARGET') -and ($Text -match 'Restore-ControllerEnvironment')
    lock_cleanup_wraps_environment_initialization = ($Text -match 'Acquire-RunLock\s+try \{\s+Initialize-ControllerEnvironment') -and ($Text -match 'finally \{\s+try \{\s+Release-RunLock')
}

$FailedStatic = @(
    $StaticChecks.GetEnumerator() | Where-Object { -not $_.Value }
)
if ($FailedStatic.Count -gt 0) {
    throw ("Controller static contract failed: {0}" -f ($FailedStatic.Name -join ', '))
}

$PythonInventory = @()
try { $PythonInventory += (& py "-0p" 2>&1 | Out-String).Trim() } catch { }
try { $PythonInventory += (& python --version 2>&1 | Out-String).Trim() } catch { }

[pscustomobject]@{
    status = "PASS"
    controller = $ControllerPath
    controller_sha256 = (Get-FileHash -LiteralPath $ControllerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    repo_root = $RepoRoot
    powershell_version = $PSVersionTable.PSVersion.ToString()
    ps_edition = $PSVersionTable.PSEdition
    controller_token_count = $ControllerParse.tokens.Count
    preflight_token_count = $SelfParse.tokens.Count
    static_checks = $StaticChecks
    python_inventory = $PythonInventory
    nvidia_smi_available = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
} | ConvertTo-Json -Depth 8
