[CmdletBinding()]
param(
    [string]$TaskName = "MumuAiSupport",
    [string]$WatchdogTaskName = "MumuAiSupportWatchdog",
    [switch]$Preview
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$statePath = Join-Path $projectRoot ".runtime\supervisor-state.json"
$supervisorPath = Join-Path $PSScriptRoot "mumu_runtime_supervisor.py"
$legacyLauncherPath = Join-Path $PSScriptRoot "start-mumu-local.ps1"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
. (Join-Path $PSScriptRoot "mumu-runtime-helpers.ps1")

$state = $null
if (Test-Path -LiteralPath $statePath -PathType Leaf) {
    try { $state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json } catch { }
}
$processInfo = Get-MumuRuntimeProcessInfo `
    -Processes @(Get-CimInstance Win32_Process) `
    -SupervisorState $state `
    -SupervisorPath $supervisorPath `
    -LegacyLauncherPath $legacyLauncherPath `
    -VenvPython $venvPython
$isRunning = @($processInfo.Supervisors).Count -gt 0 -or $null -ne $processInfo.StateChild

if ($isRunning) {
    if (-not $Preview) {
        & (Join-Path $PSScriptRoot "stop-mumu-local.ps1") -TaskName $TaskName -WaitSeconds 20 *> $null
        Disable-ScheduledTask -TaskName $TaskName | Out-Null
        Disable-ScheduledTask -TaskName $WatchdogTaskName | Out-Null
    }
    [pscustomobject]@{ TaskName = $TaskName; Action = "stopped"; Preview = [bool]$Preview } | ConvertTo-Json -Compress
    exit 0
}

if (-not $Preview) {
    # The switch owns both tasks, so a paused bot cannot be restarted by the
    # watchdog independently.
    Enable-ScheduledTask -TaskName $TaskName -ErrorAction Stop | Out-Null
    Enable-ScheduledTask -TaskName $WatchdogTaskName -ErrorAction Stop | Out-Null
    Start-ScheduledTask -TaskName $TaskName
}
[pscustomobject]@{ TaskName = $TaskName; Action = "started"; Preview = [bool]$Preview } | ConvertTo-Json -Compress
