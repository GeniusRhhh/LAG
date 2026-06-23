param(
    [string]$PythonExe = "C:\Users\ZRF\.conda\envs\qwen_poetry\python.exe",
    [string]$OutputRoot = "",
    [int]$CollectEpisodes = 4,
    [int]$CollectSteps = 4200,
    [int]$Epochs = 20,
    [int]$EvalSteps = 4200,
    [int]$OnlineUpdates = 20,
    [int]$OnlineBufferSize = 256,
    [ValidateSet("legacy", "native")]
    [string]$TeacherMode = "legacy",
    [string]$NativeModelPath = "",
    [string]$NativeMetaPath = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

$Pipeline = Join-Path $RepoRoot "scripts\tacticalProject\cap\train_enemy_f16_cap_pipeline.py"
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepoRoot "scripts\tacticalProject\models\enemy_f16_cap_full_pipeline"
}

$Args = @(
    $Pipeline,
    "--output-root", $OutputRoot,
    "--collect-episodes", $CollectEpisodes,
    "--collect-steps", $CollectSteps,
    "--teacher-mode", $TeacherMode,
    "--enable-safe-teacher",
    "--epochs", $Epochs,
    "--steps", $EvalSteps,
    "--online-updates", $OnlineUpdates,
    "--online-buffer-size", $OnlineBufferSize,
    "--online-max-steps", $EvalSteps,
    "--online-enable-safe-teacher-for-other-enemies",
    "--online-eval-steps", $EvalSteps
)

if (-not [string]::IsNullOrWhiteSpace($NativeModelPath)) {
    $Args += @("--native-model-path", $NativeModelPath)
}

if (-not [string]::IsNullOrWhiteSpace($NativeMetaPath)) {
    $Args += @("--native-meta-path", $NativeMetaPath)
}

Write-Host "[train] enemy F16 CAP full pipeline starting..."
Write-Host "[train] python: $PythonExe"
Write-Host "[train] output: $OutputRoot"
Write-Host "[train] teacher: $TeacherMode"

& $PythonExe @Args
