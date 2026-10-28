from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .runtime_state import atomic_write_json, read_json_object


MUTEX_NAME = r"Local\MumuAiSupportSupervisor"
POLL_SECONDS = 2
STREAM_GRACE_SECONDS = 120
RESTART_DELAYS_SECONDS = (5, 10, 30, 60)
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
ERROR_ALREADY_EXISTS = 183

REQUIRED_ENVIRONMENT = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "DINGTALK_CLIENT_ID",
    "DINGTALK_CLIENT_SECRET",
    "DINGTALK_ROBOT_CODE",
    "DWS_PROFILE",
    "MUMU_OPERATOR_USER_ID",
    "MUMU_ALLOWED_USER_IDS",
)

OPTIONAL_ENVIRONMENT = {
    "MUMU_SKILL_VERSION": "v2",
    "MUMU_RETRIEVAL_MODE": "legacy",
    "MUMU_KNOWLEDGE_REFRESH_SECONDS": "300",
    "MUMU_FULL_SCAN_BATCH_SIZE": "40",
    "MUMU_INTERPRETATION_ENABLED": "false",
    "DEEPSEEK_TIMEOUT_SECONDS": "45",
    "MUMU_CASE_WORKFLOW_ENABLED": "false",
    "MUMU_OWNER_USER_ID": "",
    "MUMU_OWNER_NAME": "",
    "MUMU_CASE_MAX_MESSAGES": "500",
}

PROXY_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


class RuntimeConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    venv_python: Path
    venv_site_packages: Path
    runtime_dir: Path
    stdout_log: Path
    stderr_log: Path
    stream_health: Path
    supervisor_state: Path

    @classmethod
    def for_project(cls, project_root: Path) -> "RuntimePaths":
        runtime_dir = project_root / ".runtime"
        return cls(
            project_root=project_root,
            venv_python=project_root / ".venv" / "Scripts" / "python.exe",
            venv_site_packages=project_root / ".venv" / "Lib" / "site-packages",
            runtime_dir=runtime_dir,
            stdout_log=runtime_dir / "mumu-bot-task.stdout.log",
            stderr_log=runtime_dir / "mumu-bot-task.stderr.log",
            stream_health=runtime_dir / "stream-health.json",
            supervisor_state=runtime_dir / "supervisor-state.json",
        )


def load_child_environment(
    user_environment: Mapping[str, str],
    *,
    base_environment: Mapping[str, str] | None = None,
    run_id: str | None = None,
) -> dict[str, str]:
    missing = [
        name
        for name in REQUIRED_ENVIRONMENT
        if not str(user_environment.get(name, "")).strip()
    ]
    if missing:
        raise RuntimeConfigurationError(
            "Missing user environment variables: " + ", ".join(missing)
        )

    child: dict[str, str] = {}
    path_values: list[str] = []
    sources = (
        os.environ if base_environment is None else base_environment,
        user_environment,
    )
    for source in sources:
        for name, value in source.items():
            if value is None:
                continue
            if name.lower() == "path":
                path_values.append(str(value))
                continue
            canonical_name = name.upper() if os.name == "nt" else name
            child[canonical_name] = str(value)
    if path_values:
        merged_path: list[str] = []
        seen_path_parts: set[str] = set()
        for value in reversed(path_values):
            for part in value.split(os.pathsep):
                normalized_part = part.casefold()
                if part and normalized_part not in seen_path_parts:
                    merged_path.append(part)
                    seen_path_parts.add(normalized_part)
        child["PATH"] = os.pathsep.join(merged_path)
    for name, default in OPTIONAL_ENVIRONMENT.items():
        if not child.get(name, "").strip():
            child[name] = default
    for name in PROXY_NAMES:
        child.pop(name, None)
    child["MUMU_RUNTIME_RUN_ID"] = run_id or uuid.uuid4().hex
    child["PYTHONUTF8"] = "1"
    child["PYTHONIOENCODING"] = "utf-8"
    return child


def read_user_environment() -> dict[str, str]:
    if os.name != "nt":
        raise RuntimeConfigurationError("Windows user environment is required")
    import winreg

    values: dict[str, str] = {}
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        index = 0
        while True:
            try:
                name, value, value_type = winreg.EnumValue(key, index)
            except OSError:
                break
            index += 1
            text = str(value)
            if value_type == winreg.REG_EXPAND_SZ:
                text = os.path.expandvars(text)
            values[name] = text
    return values


class ConnectionWatchdog:
    def __init__(self, run_id: str, grace_seconds: int = STREAM_GRACE_SECONDS) -> None:
        self.run_id = run_id
        self.grace_seconds = grace_seconds
        self.not_connected_since: datetime | None = None

    def observe(
        self,
        health: Mapping[str, object] | None,
        now: datetime,
    ) -> bool:
        connected = (
            health is not None
            and health.get("run_id") == self.run_id
            and health.get("state") == "connected"
        )
        if connected and health is not None:
            heartbeat = health.get("last_heartbeat_at") or health.get("updated_at")
            if heartbeat:
                try:
                    heartbeat_at = datetime.fromisoformat(str(heartbeat))
                    connected = (
                        now - heartbeat_at
                    ).total_seconds() < self.grace_seconds
                except (TypeError, ValueError):
                    connected = False
        if connected:
            self.not_connected_since = None
            return False
        if self.not_connected_since is None:
            self.not_connected_since = now
            return False
        return (now - self.not_connected_since).total_seconds() >= self.grace_seconds


class WindowsApi(Protocol):
    def create_mutex(self, name: str) -> tuple[int, bool]: ...

    def release_mutex(self, handle: int) -> None: ...

    def create_kill_on_close_job(self) -> int: ...

    def assign_process(self, job_handle: int, process_handle: int) -> None: ...

    def close_handle(self, handle: int) -> None: ...


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class CtypesWindowsApi:
    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("Windows runtime is required")
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel32.CreateMutexW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_bool,
            ctypes.c_wchar_p,
        ]
        self.kernel32.CreateMutexW.restype = ctypes.c_void_p
        self.kernel32.ReleaseMutex.argtypes = [ctypes.c_void_p]
        self.kernel32.ReleaseMutex.restype = ctypes.c_bool
        self.kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        self.kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        self.kernel32.SetInformationJobObject.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self.kernel32.SetInformationJobObject.restype = ctypes.c_bool
        self.kernel32.AssignProcessToJobObject.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.kernel32.AssignProcessToJobObject.restype = ctypes.c_bool
        self.kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self.kernel32.CloseHandle.restype = ctypes.c_bool

    @staticmethod
    def _error(operation: str) -> RuntimeError:
        return RuntimeError(f"Windows operation failed: {operation}")

    def create_mutex(self, name: str) -> tuple[int, bool]:
        handle = self.kernel32.CreateMutexW(None, True, name)
        if not handle:
            raise self._error("CreateMutexW")
        return int(handle), ctypes.get_last_error() == ERROR_ALREADY_EXISTS

    def release_mutex(self, handle: int) -> None:
        if not self.kernel32.ReleaseMutex(ctypes.c_void_p(handle)):
            raise self._error("ReleaseMutex")

    def create_kill_on_close_job(self) -> int:
        handle = self.kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise self._error("CreateJobObjectW")
        information = _ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not self.kernel32.SetInformationJobObject(
            ctypes.c_void_p(handle),
            9,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            self.kernel32.CloseHandle(ctypes.c_void_p(handle))
            raise self._error("SetInformationJobObject")
        return int(handle)

    def assign_process(self, job_handle: int, process_handle: int) -> None:
        if not self.kernel32.AssignProcessToJobObject(
            ctypes.c_void_p(job_handle),
            ctypes.c_void_p(process_handle),
        ):
            if ctypes.get_last_error() == 5:
                raise PermissionError("Windows operation denied: AssignProcessToJobObject")
            raise self._error("AssignProcessToJobObject")

    def close_handle(self, handle: int) -> None:
        if not self.kernel32.CloseHandle(ctypes.c_void_p(handle)):
            raise self._error("CloseHandle")


class WindowsNamedMutex:
    def __init__(self, name: str, *, api: WindowsApi | None = None) -> None:
        self.api = api or CtypesWindowsApi()
        self.handle, already_exists = self.api.create_mutex(name)
        self.acquired = not already_exists

    def __enter__(self) -> "WindowsNamedMutex":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if self.acquired:
                self.api.release_mutex(self.handle)
        finally:
            self.api.close_handle(self.handle)


class WindowsKillOnCloseJob:
    def __init__(self, *, api: WindowsApi | None = None) -> None:
        self.api = api or CtypesWindowsApi()
        self.handle = self.api.create_kill_on_close_job()
        self.closed = False

    def assign(self, process_handle: int) -> None:
        self.api.assign_process(self.handle, process_handle)

    def close(self) -> None:
        if not self.closed:
            self.api.close_handle(self.handle)
            self.closed = True


def _state_payload(
    *,
    run_id: str,
    state: str,
    now: datetime,
    child_pid: int = 0,
    exit_code: int | None = None,
    error_type: str = "",
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "state": state,
        "updated_at": now.isoformat(),
        "supervisor_pid": os.getpid(),
        "child_pid": child_pid,
        "exit_code": exit_code,
        "error_type": error_type,
    }


def run_supervisor(
    project_root: Path,
    *,
    user_environment: Mapping[str, str] | None = None,
    base_environment: Mapping[str, str] | None = None,
    mutex_factory: Callable[[], object] | None = None,
    job_factory: Callable[[], object] | None = None,
    process_factory: Callable[..., object] = subprocess.Popen,
    state_writer: Callable[[Path, Mapping[str, object]], None] = atomic_write_json,
    health_reader: Callable[[Path], dict[str, object] | None] = read_json_object,
    clock: Callable[[], datetime] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    run_id_factory: Callable[[], str] | None = None,
    python_executable: Path | None = None,
) -> int:
    paths = RuntimePaths.for_project(project_root)
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    mutex_builder = mutex_factory or (lambda: WindowsNamedMutex(MUTEX_NAME))
    job_builder = job_factory or WindowsKillOnCloseJob
    now_func = clock or (lambda: datetime.now().astimezone())
    run_id = (run_id_factory or (lambda: uuid.uuid4().hex))()

    with mutex_builder() as mutex:
        if not getattr(mutex, "acquired", False):
            return 0
        source_environment = (
            read_user_environment() if user_environment is None else user_environment
        )
        child_environment = load_child_environment(
            source_environment,
            base_environment=base_environment,
            run_id=run_id,
        )
        runtime_python = Path(sys.executable) if python_executable is None else python_executable
        if not runtime_python.is_file():
            raise RuntimeConfigurationError("Base Python executable is missing")
        python_paths = [
            str(paths.project_root / "src"),
            str(paths.venv_site_packages),
        ]
        existing_python_path = child_environment.get("PYTHONPATH", "").strip()
        if existing_python_path:
            python_paths.append(existing_python_path)
        child_environment["PYTHONPATH"] = os.pathsep.join(python_paths)

        with (
            paths.stdout_log.open("ab", buffering=0) as stdout_handle,
            paths.stderr_log.open("ab", buffering=0) as stderr_handle,
        ):
            process = process_factory(
                [str(runtime_python), "-m", "mumu_bot.cli", "start"],
                cwd=str(paths.project_root),
                env=child_environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                stdin=subprocess.DEVNULL,
                shell=False,
            )
            child_pid = int(getattr(process, "pid"))
            process_handle = int(getattr(process, "_handle"))
            job = job_builder()
            watchdog = ConnectionWatchdog(run_id)
            try:
                try:
                    job.assign(process_handle)
                except PermissionError:
                    # A process launched by Task Scheduler can already belong to a
                    # restrictive job. Keep supervising it when Windows rejects a
                    # nested job assignment instead of terminating a healthy bot.
                    pass
                except OSError as error:
                    if getattr(error, "winerror", None) != 5:
                        raise
                state_writer(
                    paths.supervisor_state,
                    _state_payload(
                        run_id=run_id,
                        state="running",
                        now=now_func(),
                        child_pid=child_pid,
                    ),
                )
                while True:
                    exit_code = process.poll()
                    if exit_code is not None:
                        state_writer(
                            paths.supervisor_state,
                            _state_payload(
                                run_id=run_id,
                                state="child_exited",
                                now=now_func(),
                                child_pid=child_pid,
                                exit_code=int(exit_code),
                            ),
                        )
                        return 1
                    if watchdog.observe(
                        health_reader(paths.stream_health),
                        now_func(),
                    ):
                        state_writer(
                            paths.supervisor_state,
                            _state_payload(
                                run_id=run_id,
                                state="stream_timeout",
                                now=now_func(),
                                child_pid=child_pid,
                                error_type="StreamConnectionTimeout",
                            ),
                        )
                        return 1
                    sleeper(POLL_SECONDS)
            finally:
                job.close()


def run_with_restarts(
    run_once: Callable[[], int],
    *,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    """Keep the local service available after a child or stream failure.

    A Scheduled Task restart is only a secondary safety net: it has not been
    reliable after a long-running task exits on some Windows installations.
    Configuration errors are returned by ``main`` before reaching this loop.
    """
    failures = 0
    while True:
        result = run_once()
        if result != 1:
            return result
        delay = RESTART_DELAYS_SECONDS[min(failures, len(RESTART_DELAYS_SECONDS) - 1)]
        failures += 1
        sleeper(delay)


def main(project_root: Path | None = None) -> int:
    if project_root is None:
        parser = argparse.ArgumentParser(prog="mumu-runtime-supervisor")
        parser.add_argument("--project-root", required=True)
        args = parser.parse_args()
        project_root = Path(args.project_root)
    try:
        resolved_root = project_root.resolve()
        return run_with_restarts(lambda: run_supervisor(resolved_root))
    except RuntimeConfigurationError:
        return 2
    except Exception:
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
