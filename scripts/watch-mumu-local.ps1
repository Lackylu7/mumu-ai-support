[CmdletBinding()]
param(
    [string]$TaskName = "MumuAiSupport",
    [int]$MaximumHeartbeatAgeSeconds = 180,
    [int]$MinimumRestartIntervalSeconds = 300,
    [switch]$Preview
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeDirectory = Join-Path $projectRoot ".runtime"
$statePath = Join-Path $runtimeDirectory "supervisor-state.json"
$healthPath = Join-Path $runtimeDirectory "stream-health.json"
$restartMarkerPath = Join-Path $runtimeDirectory "watchdog-last-restart.json"
$supervisorPath = Join-Path $PSScriptRoot "mumu_runtime_supervisor.py"
$legacyLauncherPath = Join-Path $PSScriptRoot "start-mumu-local.ps1"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
. (Join-Path $PSScriptRoot "mumu-runtime-helpers.ps1")

function Read-MumuJson {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    try { return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json } catch { return $null }
}

$state = Read-MumuJson -Path $statePath
$health = Read-MumuJson -Path $healthPath
$processInfo = Get-MumuRuntimeProcessInfo `
    -Processes @(Get-CimInstance Win32_Process) `
    -SupervisorState $state `
    -SupervisorPath $supervisorPath `
    -LegacyLauncherPath $legacyLauncherPath `
    -VenvPython $venvPython
$healthy = Test-MumuRuntimeHealthy `
    -ProcessInfo $processInfo `
    -SupervisorState $state `
    -StreamHealth $health `
    -MaximumHeartbeatAgeSeconds $MaximumHeartbeatAgeSeconds

$result = [ordered]@{
    TaskName = $TaskName
    Healthy = $healthy
    SupervisorProcessCount = @($processInfo.Supervisors).Count
    BotProcessPresent = [bool]$processInfo.StateChild
    StreamState = if ($health) { [string]$health.state } else { "missing" }
    Restarted = $false
    Reason = "healthy"
}
if ($healthy) {
    if ($Preview) { $result | ConvertTo-Json -Compress }
    exit 0
}

$result.Reason = "runtime_or_stream_unhealthy"
$lastRestart = Read-MumuJson -Path $restartMarkerPath
if ($lastRestart -and $lastRestart.restarted_at) {
    try {
        $lastRestartAt = [datetime]::Parse([string]$lastRestart.restarted_at)
        if (((Get-Date).ToUniversalTime() - $lastRestartAt.ToUniversalTime()).TotalSeconds -lt $MinimumRestartIntervalSeconds) {
            $result.Reason = "restart_throttled"
            if ($Preview) { $result | ConvertTo-Json -Compress }
            exit 0
        }
    } catch { }
}
if ($Preview) { $result | ConvertTo-Json -Compress; exit 0 }

New-Item -ItemType Directory -Path $runtimeDirectory -Force | Out-Null
[ordered]@{ restarted_at = (Get-Date).ToString("o") } | ConvertTo-Json -Compress | Set-Content -LiteralPath $restartMarkerPath -Encoding UTF8

& (Join-Path $PSScriptRoot "stop-mumu-local.ps1") -TaskName $TaskName -WaitSeconds 20 *> $null
Start-ScheduledTask -TaskName $TaskName
$result.Restarted = $true
$result | ConvertTo-Json -Compress
