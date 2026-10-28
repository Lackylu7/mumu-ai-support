[CmdletBinding()]
param(
    [string]$TaskName = "MumuAiSupport",
    [string]$WatchdogTaskName = "MumuAiSupportWatchdog",
    [switch]$Preview
)

$ErrorActionPreference = "Stop"

$launcherPath = Join-Path $PSScriptRoot "start-mumu-local.ps1"
if (-not (Test-Path -LiteralPath $launcherPath -PathType Leaf)) {
    throw "Launcher script not found: $launcherPath"
}
$supervisorPath = Join-Path $PSScriptRoot "mumu_runtime_supervisor.py"
if (-not (Test-Path -LiteralPath $supervisorPath -PathType Leaf)) {
    throw "Supervisor script not found: $supervisorPath"
}
$watchdogPath = Join-Path $PSScriptRoot "watch-mumu-local.ps1"
if (-not (Test-Path -LiteralPath $watchdogPath -PathType Leaf)) {
    throw "Watchdog script not found: $watchdogPath"
}
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "Virtual environment Python not found: $venvPython"
}
$basePython = (& $venvPython -c "import sys; print(sys._base_executable)").Trim()
if (-not [System.IO.Path]::IsPathRooted($basePython) -or -not (Test-Path -LiteralPath $basePython -PathType Leaf)) {
    throw "Base Python executable is invalid"
}
$runtimePython = Join-Path (Split-Path -Parent $basePython) "pythonw.exe"
if (-not (Test-Path -LiteralPath $runtimePython -PathType Leaf)) {
    throw "Windowless Python executable not found: $runtimePython"
}

$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$arguments = '-u "{0}"' -f $supervisorPath

$action = New-ScheduledTaskAction `
    -Execute $runtimePython `
    -Argument $arguments `
    -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $currentUser
$principal = New-ScheduledTaskPrincipal `
    -UserId $currentUser `
    -LogonType Interactive `
    -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -Hidden

$task = New-ScheduledTask `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Mumu DingTalk bot local runtime"

$watchdogAction = New-ScheduledTaskAction `
    -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -Argument ('-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "{0}" -TaskName "{1}"' -f $watchdogPath, $TaskName) `
    -WorkingDirectory $projectRoot
$watchdogTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At ((Get-Date).AddMinutes(1)) `
    -RepetitionInterval (New-TimeSpan -Minutes 2) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$watchdogSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew `
    -Hidden
$watchdogTask = New-ScheduledTask `
    -Action $watchdogAction `
    -Trigger $watchdogTrigger `
    -Principal $principal `
    -Settings $watchdogSettings `
    -Description "Mumu DingTalk bot availability watchdog"

if ($Preview) {
    [pscustomobject]@{
        TaskName = $TaskName
        Execute = $runtimePython
        Arguments = $arguments
        WorkingDirectory = $projectRoot
        RestartCount = 999
        RestartIntervalMinutes = 1
        WatchdogTaskName = $WatchdogTaskName
        WatchdogIntervalMinutes = 2
        ExecutionTimeLimitSeconds = 0
        MultipleInstances = "IgnoreNew"
    } | ConvertTo-Json -Compress
    exit 0
}

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
Register-ScheduledTask -TaskName $WatchdogTaskName -InputObject $watchdogTask -Force | Out-Null
Write-Output "Registered task: $TaskName"
Write-Output "Registered watchdog: $WatchdogTaskName"
