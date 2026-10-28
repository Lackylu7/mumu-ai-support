[CmdletBinding()]
param(
    [string]$TaskName = "MumuAiSupport",
    [switch]$Json,
    [switch]$ProbeModel
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$streamHealthPath = Join-Path $projectRoot ".runtime\stream-health.json"
$supervisorStatePath = Join-Path $projectRoot ".runtime\supervisor-state.json"
$supervisorPath = Join-Path $PSScriptRoot "mumu_runtime_supervisor.py"
$legacyLauncherPath = Join-Path $PSScriptRoot "start-mumu-local.ps1"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
. (Join-Path $PSScriptRoot "mumu-runtime-helpers.ps1")
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
$taskInfo = Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction Stop
$allProcesses = @(Get-CimInstance Win32_Process)
$streamHealth = $null
$supervisorState = $null
if (Test-Path -LiteralPath $streamHealthPath -PathType Leaf) {
    try { $streamHealth = Get-Content -LiteralPath $streamHealthPath -Raw -Encoding UTF8 | ConvertFrom-Json } catch { $streamHealth = $null }
}
if (Test-Path -LiteralPath $supervisorStatePath -PathType Leaf) {
    try { $supervisorState = Get-Content -LiteralPath $supervisorStatePath -Raw -Encoding UTF8 | ConvertFrom-Json } catch { $supervisorState = $null }
}
$childProcess = $null
$processInfo = Get-MumuRuntimeProcessInfo `
    -Processes $allProcesses `
    -SupervisorState $supervisorState `
    -SupervisorPath $supervisorPath `
    -LegacyLauncherPath $legacyLauncherPath `
    -VenvPython $venvPython
$childProcess = $processInfo.StateChild
$logicalInstanceCount = if ($childProcess -or $processInfo.LegacyVenv.Count -gt 0) { 1 } else { 0 }
$result = [ordered]@{
    TaskName = $task.TaskName
    TaskState = $task.State
    LastRunTime = $taskInfo.LastRunTime
    LastTaskResult = $taskInfo.LastTaskResult
    SupervisorProcessCount = $processInfo.Supervisors.Count
    BotLogicalInstanceCount = $logicalInstanceCount
    RunId = if ($supervisorState) { [string]$supervisorState.run_id } else { "" }
    StreamRunMatches = [bool]($streamHealth -and $supervisorState -and $streamHealth.run_id -eq $supervisorState.run_id)
    StreamState = if ($streamHealth) { [string]$streamHealth.state } else { "missing" }
    StreamUpdatedAt = if ($streamHealth) { [string]$streamHealth.updated_at } else { "" }
    ModelProbeSucceeded = $null
    ModelProbeMilliseconds = $null
}
if ($ProbeModel) {
    $requiredEnvironment = @(
        "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL",
        "DINGTALK_CLIENT_ID", "DINGTALK_CLIENT_SECRET", "DINGTALK_ROBOT_CODE",
        "DWS_PROFILE", "MUMU_OPERATOR_USER_ID", "MUMU_ALLOWED_USER_IDS"
    )
    foreach ($name in $requiredEnvironment) {
        $value = [Environment]::GetEnvironmentVariable($name, "User")
        if ([string]::IsNullOrWhiteSpace($value)) { throw "Missing user environment variable: $name" }
        Set-Item -LiteralPath "Env:$name" -Value $value
    }
    foreach ($name in @("MUMU_RETRIEVAL_MODE", "MUMU_KNOWLEDGE_REFRESH_SECONDS", "MUMU_FULL_SCAN_BATCH_SIZE", "MUMU_DOMAIN_ROUTING_ENABLED", "MUMU_DOMAIN_ROUTING_MIN_CONFIDENCE", "DEEPSEEK_TIMEOUT_SECONDS", "MUMU_CASE_WORKFLOW_ENABLED", "MUMU_OWNER_USER_ID", "MUMU_OWNER_NAME", "MUMU_CASE_MAX_MESSAGES")) {
        $value = [Environment]::GetEnvironmentVariable($name, "User")
        if (-not [string]::IsNullOrWhiteSpace($value)) { Set-Item -LiteralPath "Env:$name" -Value $value }
    }
    foreach ($name in @("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")) { Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue }
    $start = Get-Date
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $probeExitCode = 1
    Push-Location -LiteralPath $projectRoot
    try {
        & $venvPython -m mumu_bot.cli ask --question "Reply only: healthy" --user-id "local-health-probe" *> $null
        $probeExitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
        $ErrorActionPreference = $previousErrorAction
    }
    $result.ModelProbeSucceeded = ($probeExitCode -eq 0)
    $result.ModelProbeMilliseconds = [int]((Get-Date) - $start).TotalMilliseconds
}
if ($Json) { $result | ConvertTo-Json -Compress } else { [pscustomobject]$result | Format-List }
