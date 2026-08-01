$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$runtimeRoot = if ($env:DUBBING_STUDIO_RUNTIME) { $env:DUBBING_STUDIO_RUNTIME } else { Join-Path $projectRoot 'runtime' }
$portableFfmpeg = Get-ChildItem -LiteralPath (Join-Path $runtimeRoot 'tools\ffmpeg') -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if ($portableFfmpeg) { $env:Path = "$($portableFfmpeg.Directory.FullName);$env:Path" }

if (-not (Test-Path -LiteralPath $python)) {
    throw 'Dubbing Studio не установлен. Запустите setup.bat и дождитесь сообщения об успешной установке.'
}

$env:PYTHONUTF8 = '1'
$doctor = Join-Path $projectRoot 'tools\doctor.py'
& $python $doctor
if ($LASTEXITCODE -ne 0) { throw 'Не все компоненты готовы. Повторно запустите setup.bat.' }

$occupied = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
if ($occupied) {
    Start-Process 'http://127.0.0.1:8765' | Out-Null
    Write-Host 'Dubbing Studio уже запущен; открыта существующая вкладка.' -ForegroundColor Yellow
    exit 0
}

Start-Job -ScriptBlock { Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8765' } | Out-Null
Write-Host 'Dubbing Studio: http://127.0.0.1:8765' -ForegroundColor Green
Write-Host 'Чтобы остановить сервер, закройте это окно или нажмите Ctrl+C.'
& $python -m uvicorn app.main:app --host 127.0.0.1 --port 8765 --app-dir $projectRoot



