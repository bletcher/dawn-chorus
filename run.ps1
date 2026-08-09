<#
.SYNOPSIS
  Dawn-chorus local workflow. Wraps tools/run.py using the project's own venv, so nothing
  depends on which python happens to be on PATH.

.EXAMPLE
  .\run.ps1 setup                      # one-time: create .venv and install the run stack
  .\run.ps1 status                     # what's configured, what's unprocessed
  .\run.ps1 all                        # inference on new recordings, then rebuild dashboards
  .\run.ps1 process -Site montague -Deployment owl
  .\run.ps1 webapp                     # control panel at http://127.0.0.1:8765
  .\run.ps1 compare                    # two recorders at one site, head to head
  .\run.ps1 serve                      # http://127.0.0.1:8000/site/
  .\run.ps1 publish -- --push          # regenerate public JSON and deploy

.NOTES
  Arguments after the command are passed straight through to tools/run.py, so any flag it
  accepts works here. Use `--` before flags PowerShell would otherwise try to interpret.
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'status', 'process', 'dashboard', 'compare', 'publish', 'serve', 'webapp', 'all', 'test')]
    [string]$Command = 'status',

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $root '.venv\Scripts\python.exe'

function Invoke-Setup {
    Write-Host 'creating .venv and installing the run stack (LiteRT, not TensorFlow)...'
    $sys = (Get-Command py -ErrorAction SilentlyContinue)
    if ($sys) { & py -3 -m venv (Join-Path $root '.venv') }
    else { & python -m venv (Join-Path $root '.venv') }
    & $py -m pip install --upgrade pip --quiet
    & $py -m pip install -r (Join-Path $root 'requirements-run.txt')
    & $py -m pip install -e $root --no-deps
    & $py -c "import ai_edge_litert, librosa, dawnchorus; print('setup OK')"
    Write-Host ''
    Write-Host 'Next:  .\run.ps1 status'
}

if ($Command -eq 'setup') { Invoke-Setup; return }

if (-not (Test-Path $py)) {
    Write-Error "no venv found at $py`nRun:  .\run.ps1 setup"
    exit 1
}

if ($Command -eq 'test') {
    & $py -m pytest (Join-Path $root 'tests') -q @Rest
    exit $LASTEXITCODE
}

# Strip a leading literal '--' so `.\run.ps1 publish -- --push` reads naturally.
if ($Rest -and $Rest[0] -eq '--') { $Rest = $Rest[1..($Rest.Length - 1)] }

& $py (Join-Path $root 'tools\run.py') $Command @Rest
exit $LASTEXITCODE
