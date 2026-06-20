<#
.SYNOPSIS
    Sarvam TTS Dataset Pipeline Runner (Windows / PowerShell)
#>

[CmdletBinding()]
param(
    [string] $SourcesFile = "data/sources_en.yaml",
    [int]    $SmallBatch  = 0,
    [string] $Language    = "en-IN",
    [ValidateSet("download","preprocess","segment","quality","transcribe","diarize","emotion")]
    [string] $StageFrom   = "download",
    [switch] $DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-OK   { param([string]$Msg) Write-Host "OK  $Msg" -ForegroundColor Green }
function Write-Warn { param([string]$Msg) Write-Host "!!  $Msg" -ForegroundColor Yellow }
function Write-Fail { param([string]$Msg) Write-Host "ERR $Msg" -ForegroundColor Red }
function Write-Info { param([string]$Msg) Write-Host "->  $Msg" -ForegroundColor Gray }

if (-not (Test-Path "pyproject.toml")) {
    Write-Fail "Run this script from the project root (where pyproject.toml lives)."
    exit 1
}

if (-not (Test-Path $SourcesFile)) {
    Write-Fail "Sources file not found: $SourcesFile"
    exit 1
}

if (-not (Test-Path ".env")) {
    Write-Fail ".env file not found. Copy .env.template to .env and fill in credentials."
    exit 1
}

try {
    $ffmpegVer = & ffmpeg -version 2>&1 | Select-Object -First 1
    Write-OK "ffmpeg found: $ffmpegVer"
} catch {
    Write-Fail "ffmpeg not found on PATH. Install it, then restart this terminal."
    exit 1
}

try {
    $uvVer = & uv --version 2>&1
    Write-OK "uv found: $uvVer"
} catch {
    Write-Fail "uv not found. Install from https://docs.astral.sh/uv/"
    exit 1
}

$pipelineRoot = Get-Location
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$pipelineRoot;$env:PYTHONPATH" } else { $pipelineRoot }
$env:PYTHONIOENCODING = "utf-8"

$argsList = @(
    "--sources", $SourcesFile,
    "--language", $Language,
    "--stage-from", $StageFrom
)
if ($SmallBatch -gt 0) {
    $argsList += @("--limit", "$SmallBatch")
}

Write-Host ""
Write-Host "Sarvam TTS Dataset Pipeline" -ForegroundColor Cyan
Write-Info "Sources file : $SourcesFile"
Write-Info "Language     : $Language"
Write-Info "Small batch  : $(if ($SmallBatch -gt 0) { $SmallBatch } else { 'disabled' })"
Write-Info "Resume from  : $StageFrom"
Write-Info "PYTHONPATH   : $pipelineRoot"

$cmdPreview = "uv run python -m src.pipeline " + ($argsList -join " ")
Write-Info $cmdPreview

if ($DryRun) {
    Write-Warn "Dry run only; command not executed."
    exit 0
}

$startTime = Get-Date
& uv run python -m src.pipeline @argsList
$exitCode = $LASTEXITCODE
$elapsed = ((Get-Date) - $startTime).TotalSeconds

if ($exitCode -ne 0) {
    Write-Fail "Pipeline failed with exit code $exitCode after ${elapsed}s."
    Write-Warn "After fixing the error, resume with: .\the pipeline\scripts\run_pipeline.ps1 -StageFrom $StageFrom -SourcesFile $SourcesFile -Language $Language"
    exit $exitCode
}

Write-OK "Pipeline complete in ${elapsed}s."
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Info "1. uv run python the pipeline/scripts/run_audit.py"
Write-Info "2. uv run python -m src.dataset_builder"
Write-Info "3. uv run python -m src.upload_hf --repo-id <user/dataset>"
Write-Host "Next steps:" -ForegroundColor White
Write-Info "1. uv run python scripts/run_audit.py"
Write-Info "2. uv run python -m src.dataset_builder"
Write-Info "3. uv run python -m src.upload_hf --repo-id <user/dataset>"
