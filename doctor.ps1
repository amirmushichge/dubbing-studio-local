$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$RuntimeRoot = if ($env:DUBBING_STUDIO_RUNTIME) { $env:DUBBING_STUDIO_RUNTIME } else { Join-Path $ProjectRoot 'runtime' }
$PortableFfmpeg = Get-ChildItem -LiteralPath (Join-Path $RuntimeRoot 'tools\ffmpeg') -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if ($PortableFfmpeg) { $env:Path = "$($PortableFfmpeg.Directory.FullName);$env:Path" }
if (-not (Test-Path -LiteralPath $Python)) { throw 'Приложение не установлено. Запустите setup.bat.' }
& $Python (Join-Path $ProjectRoot 'tools\doctor.py') @args
if ($LASTEXITCODE -ne 0) { throw 'Диагностика обнаружила отсутствующие компоненты.' }



