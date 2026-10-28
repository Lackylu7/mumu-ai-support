[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$supervisorPath = Join-Path $PSScriptRoot "mumu_runtime_supervisor.py"

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "Mumu virtual environment not found: $venvPython"
}
if (-not (Test-Path -LiteralPath $supervisorPath -PathType Leaf)) {
    throw "Supervisor script not found: $supervisorPath"
}
$basePython = (& $venvPython -c "import sys; print(sys._base_executable)").Trim()
if (-not (Test-Path -LiteralPath $basePython -PathType Leaf)) {
    throw "Base Python executable not found"
}
Set-Location -LiteralPath $projectRoot
& $basePython -u $supervisorPath
exit $LASTEXITCODE
