[CmdletBinding()]
param(
    [string]$TaskName = "MumuAiSupport",
    [switch]$Preview,
    [int]$WaitSeconds = 15
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$supervisorPath = Join-Path $PSScriptRoot "mumu_runtime_supervisor.py"
$legacyLauncherPath = Join-Path $PSScriptRoot "start-mumu-local.ps1"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$statePath = Join-Path $projectRoot ".runtime\supervisor-state.json"
. (Join-Path $PSScriptRoot "mumu-runtime-helpers.ps1")

function Get-MumuRuntimeProcesses {
    $all = @(Get-CimInstance Win32_Process)
    $state = $null
    if (Test-Path -LiteralPath $statePath -PathType Leaf) {
        try {
            $state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
        } catch { }
    }
    $processInfo = Get-MumuRuntimeProcessInfo `
        -Processes $all `
        -SupervisorState $state `
        -SupervisorPath $supervisorPath `
        -LegacyLauncherPath $legacyLauncherPath `
        -VenvPython $venvPython
    return @($processInfo.StopTargets)
}

$processes = @(Get-MumuRuntimeProcesses)
$summary = [ordered]@{
    TaskName = $TaskName
    ProjectRoot = $projectRoot
    TargetProcessCount = $processes.Count
    TargetProcessIds = @($processes | ForEach-Object { [int]$_.ProcessId })
    Preview = [bool]$Preview
}
if ($Preview) { $summary | ConvertTo-Json -Compress; exit 0 }

Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
$deadline = (Get-Date).AddSeconds($WaitSeconds)
do {
    Start-Sleep -Milliseconds 250
    $processes = @(Get-MumuRuntimeProcesses)
} while ($processes.Count -gt 0 -and (Get-Date) -lt $deadline)

$parentIds = @($processes.ParentProcessId)
$ordered = @($processes | Sort-Object @{ Expression = { if ($parentIds -contains [int]$_.ProcessId) { 1 } else { 0 } } })
foreach ($process in $ordered) { Stop-Process -Id ([int]$process.ProcessId) -Force -ErrorAction SilentlyContinue }
Start-Sleep -Milliseconds 500
$remaining = @(Get-MumuRuntimeProcesses)
$summary.FallbackTerminatedCount = $processes.Count
$summary.RemainingProcessCount = $remaining.Count
$summary | ConvertTo-Json -Compress
if ($remaining.Count -gt 0) { exit 1 }
