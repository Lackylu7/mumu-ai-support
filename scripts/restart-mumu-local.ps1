[CmdletBinding()]
param(
    [string]$TaskName = "MumuAiSupport",
    [switch]$Preview,
    [int]$WaitSeconds = 120
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "mumu-runtime-helpers.ps1")
if ($Preview) {
    [pscustomobject]@{
        TaskName = $TaskName
        ProjectRoot = $projectRoot
        WaitSeconds = $WaitSeconds
        Preview = $true
    } | ConvertTo-Json -Compress
    exit 0
}

$stopScript = Join-Path $PSScriptRoot "stop-mumu-local.ps1"
$healthPath = Join-Path $projectRoot ".runtime\stream-health.json"
$statePath = Join-Path $projectRoot ".runtime\supervisor-state.json"
$oldRunId = ""
if (Test-Path -LiteralPath $statePath -PathType Leaf) {
    try {
        $oldState = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
        $oldRunId = [string]$oldState.run_id
    } catch { $oldRunId = "" }
}
& $stopScript -TaskName $TaskName -WaitSeconds 20 *> $null
Start-ScheduledTask -TaskName $TaskName
$deadline = (Get-Date).AddSeconds($WaitSeconds)
$connected = $false
do {
    Start-Sleep -Seconds 2
    try {
        $health = Get-Content -LiteralPath $healthPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
        $connected = Test-MumuRestartReady -OldRunId $oldRunId -SupervisorState $state -StreamHealth $health
    } catch { $connected = $false }
} while (-not $connected -and (Get-Date) -lt $deadline)
[pscustomobject]@{
    TaskName = $TaskName
    StreamConnected = $connected
    WaitSeconds = $WaitSeconds
} | ConvertTo-Json -Compress
if (-not $connected) { exit 1 }
