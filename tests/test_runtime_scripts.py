import json
import os
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
RUNTIME_HELPERS = SCRIPTS / "mumu-runtime-helpers.ps1"


def run_powershell(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def run_powershell_command(
    command: str,
    *,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    child_environment = os.environ.copy()
    if environment:
        child_environment.update(environment)
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        cwd=PROJECT_ROOT,
        env=child_environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


@unittest.skipUnless(os.name == "nt", "运行时脚本只在 Windows 验收")
class RuntimeScriptTests(unittest.TestCase):
    def test_restart_readiness_requires_a_new_matching_connected_run(self) -> None:
        helper = str(RUNTIME_HELPERS).replace("'", "''")
        command = f"""
$ErrorActionPreference = 'Stop'
. '{helper}'
$old = [pscustomobject]@{{ run_id = 'old-run' }}
$new = [pscustomobject]@{{ run_id = 'new-run' }}
$empty = [pscustomobject]@{{ run_id = '' }}
$oldConnected = [pscustomobject]@{{ run_id = 'old-run'; state = 'connected' }}
$newConnected = [pscustomobject]@{{ run_id = 'new-run'; state = 'connected' }}
$newConnecting = [pscustomobject]@{{ run_id = 'new-run'; state = 'connecting' }}
@(
    (Test-MumuRestartReady -OldRunId 'old-run' -SupervisorState $old -StreamHealth $oldConnected),
    (Test-MumuRestartReady -OldRunId 'old-run' -SupervisorState $empty -StreamHealth $empty),
    (Test-MumuRestartReady -OldRunId 'old-run' -SupervisorState $new -StreamHealth $oldConnected),
    (Test-MumuRestartReady -OldRunId 'old-run' -SupervisorState $new -StreamHealth $newConnecting),
    (Test-MumuRestartReady -OldRunId 'old-run' -SupervisorState $new -StreamHealth $newConnected)
) | ConvertTo-Json -Compress
"""

        result = run_powershell_command(command)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [False, False, False, False, True])

    def test_runtime_health_requires_matching_processes_and_fresh_connection(self) -> None:
        helper = str(RUNTIME_HELPERS).replace("'", "''")
        command = f"""
$ErrorActionPreference = 'Stop'
. '{helper}'
$now = [datetime]'2026-08-26T12:00:00+08:00'
$processes = [pscustomobject]@{{ Supervisors = @([pscustomobject]@{{ ProcessId = 10 }}); StateChild = [pscustomobject]@{{ ProcessId = 11 }} }}
$state = [pscustomobject]@{{ run_id = 'run-1' }}
$fresh = [pscustomobject]@{{ run_id = 'run-1'; state = 'connected'; last_heartbeat_at = '2026-08-26T11:58:00+08:00' }}
$stale = [pscustomobject]@{{ run_id = 'run-1'; state = 'connected'; last_heartbeat_at = '2026-08-26T11:50:00+08:00' }}
@(
    (Test-MumuRuntimeHealthy -ProcessInfo $processes -SupervisorState $state -StreamHealth $fresh -Now $now),
    (Test-MumuRuntimeHealthy -ProcessInfo $processes -SupervisorState $state -StreamHealth $stale -Now $now),
    (Test-MumuRuntimeHealthy -ProcessInfo ([pscustomobject]@{{ Supervisors = @(); StateChild = $null }}) -SupervisorState $state -StreamHealth $fresh -Now $now)
) | ConvertTo-Json -Compress
"""
        result = run_powershell_command(command)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [True, False, False])

    def test_process_selection_validates_supervisor_identity_and_parent_chain(self) -> None:
        supervisor_path = str(SCRIPTS / "mumu_runtime_supervisor.py")
        launcher_path = str(SCRIPTS / "start-mumu-local.ps1")
        venv_python = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
        processes = [
            {
                "ProcessId": 100,
                "ParentProcessId": 10,
                "Name": "python.exe",
                "ExecutablePath": r"C:\Python312\python.exe",
                "CommandLine": f'"C:\\Python312\\python.exe" -u "{supervisor_path}"',
            },
            {
                "ProcessId": 101,
                "ParentProcessId": 100,
                "Name": "python.exe",
                "ExecutablePath": r"C:\Python312\python.exe",
                "CommandLine": '"C:\\Python312\\python.exe" -m mumu_bot.cli start',
            },
            {
                "ProcessId": 201,
                "ParentProcessId": 999,
                "Name": "python.exe",
                "ExecutablePath": r"C:\Python312\python.exe",
                "CommandLine": '"C:\\Python312\\python.exe" -m mumu_bot.cli start',
            },
            {
                "ProcessId": 202,
                "ParentProcessId": 20,
                "Name": "python.exe",
                "ExecutablePath": r"C:\OtherProject\.venv\Scripts\python.exe",
                "CommandLine": 'python.exe -m mumu_bot.cli start',
            },
            {
                "ProcessId": 203,
                "ParentProcessId": 20,
                "Name": "python.exe",
                "ExecutablePath": r"C:\Python312\python.exe",
                "CommandLine": f'python.exe "{supervisor_path}.bak"',
            },
            {
                "ProcessId": 301,
                "ParentProcessId": 30,
                "Name": "python.exe",
                "ExecutablePath": venv_python,
                "CommandLine": f'"{venv_python}" -m mumu_bot.cli start',
            },
            {
                "ProcessId": 302,
                "ParentProcessId": 30,
                "Name": "powershell.exe",
                "ExecutablePath": r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
                "CommandLine": f'powershell.exe -File "{launcher_path}"',
            },
            {
                "ProcessId": 303,
                "ParentProcessId": 30,
                "Name": "powershell.exe",
                "ExecutablePath": r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
                "CommandLine": f'powershell.exe -File "{launcher_path}.bak"',
            },
        ]
        fixture = {
            "processes": processes,
            "valid_state": {"supervisor_pid": 100, "child_pid": 101},
            "reused_state": {"supervisor_pid": 100, "child_pid": 201},
        }
        helper = str(RUNTIME_HELPERS).replace("'", "''")
        command = f"""
$ErrorActionPreference = 'Stop'
. '{helper}'
$fixture = $env:MUMU_RUNTIME_TEST_FIXTURE | ConvertFrom-Json
$valid = Get-MumuRuntimeProcessInfo -Processes @($fixture.processes) -SupervisorState $fixture.valid_state -SupervisorPath '{supervisor_path.replace("'", "''")}' -LegacyLauncherPath '{launcher_path.replace("'", "''")}' -VenvPython '{venv_python.replace("'", "''")}'
$reused = Get-MumuRuntimeProcessInfo -Processes @($fixture.processes) -SupervisorState $fixture.reused_state -SupervisorPath '{supervisor_path.replace("'", "''")}' -LegacyLauncherPath '{launcher_path.replace("'", "''")}' -VenvPython '{venv_python.replace("'", "''")}'
[pscustomobject]@{{
    SupervisorIds = @($valid.Supervisors | ForEach-Object {{ [int]$_.ProcessId }})
    StateChildId = if ($valid.StateChild) {{ [int]$valid.StateChild.ProcessId }} else {{ 0 }}
    StopTargetIds = @($valid.StopTargets | ForEach-Object {{ [int]$_.ProcessId }} | Sort-Object)
    ReusedStateChildId = if ($reused.StateChild) {{ [int]$reused.StateChild.ProcessId }} else {{ 0 }}
    ReusedStopTargetIds = @($reused.StopTargets | ForEach-Object {{ [int]$_.ProcessId }} | Sort-Object)
}} | ConvertTo-Json -Compress
"""

        result = run_powershell_command(
            command,
            environment={"MUMU_RUNTIME_TEST_FIXTURE": json.dumps(fixture)},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["SupervisorIds"], [100])
        self.assertEqual(payload["StateChildId"], 101)
        self.assertEqual(payload["StopTargetIds"], [100, 101, 301, 302])
        self.assertEqual(payload["ReusedStateChildId"], 0)
        self.assertEqual(payload["ReusedStopTargetIds"], [100, 301, 302])

    def test_install_preview_describes_direct_base_python_action_without_secrets(self) -> None:
        result = run_powershell(SCRIPTS / "install-mumu-local-task.ps1", "-Preview")

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["TaskName"], "MumuAiSupport")
        self.assertTrue(payload["Execute"].lower().endswith("pythonw.exe"))
        self.assertIn("mumu_runtime_supervisor.py", payload["Arguments"])
        self.assertIn(str(PROJECT_ROOT), payload["Arguments"])
        self.assertEqual(payload["RestartCount"], 999)
        self.assertEqual(payload["RestartIntervalMinutes"], 1)
        self.assertEqual(payload["WatchdogTaskName"], "MumuAiSupportWatchdog")
        self.assertEqual(payload["WatchdogIntervalMinutes"], 2)
        self.assertEqual(payload["ExecutionTimeLimitSeconds"], 0)
        self.assertEqual(payload["MultipleInstances"], "IgnoreNew")
        self.assertNotIn("sk-", result.stdout)
        self.assertNotIn("secret", result.stdout.lower())

    def test_stop_and_restart_preview_do_not_change_task(self) -> None:
        before = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "(Get-ScheduledTask -TaskName 'MumuAiSupport').State",
            ],
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        ).stdout.strip()

        stop = run_powershell(SCRIPTS / "stop-mumu-local.ps1", "-Preview")
        restart = run_powershell(SCRIPTS / "restart-mumu-local.ps1", "-Preview")

        self.assertEqual(stop.returncode, 0, stop.stderr)
        self.assertEqual(restart.returncode, 0, restart.stderr)
        self.assertEqual(
            subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    "(Get-ScheduledTask -TaskName 'MumuAiSupport').State",
                ],
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            ).stdout.strip(),
            before,
        )

    def test_toggle_preview_detects_stopped_or_running_without_changing_task(self) -> None:
        before = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "(Get-ScheduledTask -TaskName 'MumuAiSupport').State",
            ],
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        ).stdout.strip()

        result = run_powershell(SCRIPTS / "toggle-mumu-local.ps1", "-Preview")

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertIn(payload["Action"], {"started", "stopped"})
        self.assertTrue(payload["Preview"])
        after = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "(Get-ScheduledTask -TaskName 'MumuAiSupport').State",
            ],
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        ).stdout.strip()
        self.assertEqual(after, before)

    def test_toggle_controls_both_runtime_and_watchdog_tasks(self) -> None:
        source = (SCRIPTS / "toggle-mumu-local.ps1").read_text(encoding="utf-8")

        self.assertIn("Disable-ScheduledTask -TaskName $TaskName", source)
        self.assertIn("Disable-ScheduledTask -TaskName $WatchdogTaskName", source)
        self.assertIn("Enable-ScheduledTask -TaskName $WatchdogTaskName", source)

    def test_status_json_is_safe_and_does_not_dump_raw_logs(self) -> None:
        result = run_powershell(SCRIPTS / "status-mumu-local.ps1", "-Json")

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertIn("TaskState", payload)
        self.assertIn("StreamState", payload)
        self.assertNotIn("ticket", result.stdout.lower())
        self.assertNotIn("endpoint", result.stdout.lower())
        self.assertNotIn("sk-", result.stdout.lower())

    def test_model_probe_runs_from_the_project_root(self) -> None:
        source = (SCRIPTS / "status-mumu-local.ps1").read_text(encoding="utf-8")

        self.assertIn("Push-Location -LiteralPath $projectRoot", source)
        self.assertIn("Pop-Location", source)
        self.assertLess(
            source.index("Push-Location -LiteralPath $projectRoot"),
            source.index("-m mumu_bot.cli ask"),
        )


if __name__ == "__main__":
    unittest.main()
