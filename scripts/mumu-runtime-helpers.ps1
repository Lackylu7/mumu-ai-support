function Test-MumuCommandLineArgument {
    param(
        [AllowNull()][string]$CommandLine,
        [Parameter(Mandatory = $true)][string]$Argument
    )

    if ([string]::IsNullOrWhiteSpace($CommandLine) -or [string]::IsNullOrWhiteSpace($Argument)) {
        return $false
    }
    $pattern = '(?i)(?:^|[\s"])' + [regex]::Escape($Argument) + '(?=$|[\s"])'
    return [regex]::IsMatch($CommandLine, $pattern)
}

function Test-MumuBotStartProcess {
    param([Parameter(Mandatory = $true)]$Process)

    return (
        $Process.Name -match '^pythonw?\.exe$' -and
        -not [string]::IsNullOrWhiteSpace([string]$Process.CommandLine) -and
        [regex]::IsMatch(
            [string]$Process.CommandLine,
            '(?i)(?:^|\s)-m\s+mumu_bot\.cli\s+start(?:\s|$)'
        )
    )
}

function Test-MumuSupervisorProcess {
    param(
        [Parameter(Mandatory = $true)]$Process,
        [Parameter(Mandatory = $true)][string]$SupervisorPath
    )

    return (
        $Process.Name -match '^pythonw?\.exe$' -and
        (Test-MumuCommandLineArgument -CommandLine $Process.CommandLine -Argument $SupervisorPath)
    )
}

function Test-MumuLegacyPowerShellProcess {
    param(
        [Parameter(Mandatory = $true)]$Process,
        [Parameter(Mandatory = $true)][string]$LegacyLauncherPath
    )

    return (
        $Process.Name -match '^powershell(?:\.exe)?$' -and
        (Test-MumuCommandLineArgument -CommandLine $Process.CommandLine -Argument $LegacyLauncherPath)
    )
}

function Test-MumuLegacyVenvProcess {
    param(
        [Parameter(Mandatory = $true)]$Process,
        [Parameter(Mandatory = $true)][string]$VenvPython
    )

    return (
        (Test-MumuBotStartProcess -Process $Process) -and
        [string]::Equals(
            [string]$Process.ExecutablePath,
            $VenvPython,
            [System.StringComparison]::OrdinalIgnoreCase
        )
    )
}

function Get-MumuRuntimeProcessInfo {
    param(
        [Parameter(Mandatory = $true)][object[]]$Processes,
        [AllowNull()]$SupervisorState,
        [Parameter(Mandatory = $true)][string]$SupervisorPath,
        [Parameter(Mandatory = $true)][string]$LegacyLauncherPath,
        [Parameter(Mandatory = $true)][string]$VenvPython
    )

    $supervisors = @($Processes | Where-Object {
        Test-MumuSupervisorProcess -Process $_ -SupervisorPath $SupervisorPath
    })
    $legacyPowerShell = @($Processes | Where-Object {
        Test-MumuLegacyPowerShellProcess -Process $_ -LegacyLauncherPath $LegacyLauncherPath
    })
    $legacyVenv = @($Processes | Where-Object {
        Test-MumuLegacyVenvProcess -Process $_ -VenvPython $VenvPython
    })

    $stateChild = $null
    $stateSupervisorPid = 0
    $stateChildPid = 0
    if (
        $SupervisorState -and
        [int]::TryParse([string]$SupervisorState.supervisor_pid, [ref]$stateSupervisorPid) -and
        [int]::TryParse([string]$SupervisorState.child_pid, [ref]$stateChildPid) -and
        $stateSupervisorPid -gt 0 -and
        $stateChildPid -gt 0
    ) {
        $stateSupervisor = $supervisors | Where-Object {
            [int]$_.ProcessId -eq $stateSupervisorPid
        } | Select-Object -First 1
        if ($stateSupervisor) {
            $stateChild = $Processes | Where-Object {
                [int]$_.ProcessId -eq $stateChildPid -and
                [int]$_.ParentProcessId -eq $stateSupervisorPid -and
                (Test-MumuBotStartProcess -Process $_)
            } | Select-Object -First 1
        }
    }

    $targetIds = New-Object 'System.Collections.Generic.HashSet[int]'
    foreach ($process in @($supervisors) + @($legacyPowerShell) + @($legacyVenv)) {
        if ($process) { [void]$targetIds.Add([int]$process.ProcessId) }
    }
    if ($stateChild) { [void]$targetIds.Add([int]$stateChild.ProcessId) }

    [pscustomobject]@{
        Supervisors = @($supervisors)
        StateChild = $stateChild
        LegacyPowerShell = @($legacyPowerShell)
        LegacyVenv = @($legacyVenv)
        StopTargets = @($Processes | Where-Object { $targetIds.Contains([int]$_.ProcessId) })
    }
}

function Test-MumuRestartReady {
    param(
        [AllowEmptyString()][string]$OldRunId,
        [AllowNull()]$SupervisorState,
        [AllowNull()]$StreamHealth
    )

    if (-not $SupervisorState -or -not $StreamHealth) { return $false }
    $supervisorRunId = [string]$SupervisorState.run_id
    $healthRunId = [string]$StreamHealth.run_id
    return (
        -not [string]::IsNullOrWhiteSpace($supervisorRunId) -and
        -not [string]::IsNullOrWhiteSpace($healthRunId) -and
        -not [string]::Equals($supervisorRunId, $OldRunId, [System.StringComparison]::Ordinal) -and
        [string]::Equals($supervisorRunId, $healthRunId, [System.StringComparison]::Ordinal) -and
        [string]::Equals([string]$StreamHealth.state, 'connected', [System.StringComparison]::Ordinal)
    )
}

function Test-MumuRuntimeHealthy {
    param(
        [Parameter(Mandatory = $true)]$ProcessInfo,
        [AllowNull()]$SupervisorState,
        [AllowNull()]$StreamHealth,
        [int]$MaximumHeartbeatAgeSeconds = 180,
        [datetime]$Now = (Get-Date)
    )

    if (
        @($ProcessInfo.Supervisors).Count -ne 1 -or
        -not $ProcessInfo.StateChild -or
        -not $SupervisorState -or
        -not $StreamHealth
    ) {
        return $false
    }
    if (
        -not [string]::Equals([string]$SupervisorState.run_id, [string]$StreamHealth.run_id, [System.StringComparison]::Ordinal) -or
        -not [string]::Equals([string]$StreamHealth.state, 'connected', [System.StringComparison]::Ordinal)
    ) {
        return $false
    }
    $heartbeatText = [string]$StreamHealth.last_heartbeat_at
    if ([string]::IsNullOrWhiteSpace($heartbeatText)) {
        $heartbeatText = [string]$StreamHealth.updated_at
    }
    try {
        $heartbeat = [datetime]::Parse($heartbeatText)
    } catch {
        return $false
    }
    return (($Now.ToUniversalTime() - $heartbeat.ToUniversalTime()).TotalSeconds -lt $MaximumHeartbeatAgeSeconds)
}
