$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$runtimeRoot = if ($env:DUBBING_STUDIO_RUNTIME) { $env:DUBBING_STUDIO_RUNTIME } else { Join-Path $projectRoot 'runtime' }
$portableFfmpeg = Get-ChildItem -LiteralPath (Join-Path $runtimeRoot 'tools\ffmpeg') -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if ($portableFfmpeg) { $env:Path = "$($portableFfmpeg.Directory.FullName);$env:Path" }
$portableSox = Get-ChildItem -LiteralPath (Join-Path $runtimeRoot 'tools\sox') -Recurse -Filter sox.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if ($portableSox) { $env:Path = "$($portableSox.Directory.FullName);$env:Path" }

if (-not (Test-Path -LiteralPath $python)) {
    throw 'Dubbing Studio is not installed. Run setup.bat and wait for the success message.'
}

$env:PYTHONUTF8 = '1'
$doctor = Join-Path $projectRoot 'tools\doctor.py'
& $python $doctor
if ($LASTEXITCODE -ne 0) { throw 'Some components are missing. Run setup.bat again.' }

$occupied = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
if ($occupied) {
    try {
        $runningHealth = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/health' -TimeoutSec 3
    } catch {
        throw 'Port 8765 is occupied by another application. Close that application or free the port before starting Dubbing Studio.'
    }
    if ($runningHealth.app_id -ne 'dubbing-studio-local') {
        throw 'Port 8765 is occupied by another application. Dubbing Studio was not opened.'
    }
    Start-Process 'http://127.0.0.1:8765' | Out-Null
    Write-Host 'Dubbing Studio is already running; the existing workspace was opened.' -ForegroundColor Yellow
    exit 0
}

Start-Job -ScriptBlock { Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8765' } | Out-Null
Write-Host 'Dubbing Studio: http://127.0.0.1:8765' -ForegroundColor Green
Write-Host 'Close this window or press Ctrl+C to stop the server.'
& $python -m uvicorn app.main:app --host 127.0.0.1 --port 8765 --app-dir $projectRoot




