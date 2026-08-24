$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$RuntimeRoot = if ($env:DUBBING_STUDIO_RUNTIME) { $env:DUBBING_STUDIO_RUNTIME } else { Join-Path $ProjectRoot 'runtime' }
$PortableFfmpeg = Get-ChildItem -LiteralPath (Join-Path $RuntimeRoot 'tools\ffmpeg') -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if ($PortableFfmpeg) { $env:Path = "$($PortableFfmpeg.Directory.FullName);$env:Path" }
$PortableSox = Get-ChildItem -LiteralPath (Join-Path $RuntimeRoot 'tools\sox') -Recurse -Filter sox.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if ($PortableSox) { $env:Path = "$($PortableSox.Directory.FullName);$env:Path" }
if (-not (Test-Path -LiteralPath $Python)) { throw 'Dubbing Studio is not installed. Run setup.bat first.' }
& $Python (Join-Path $ProjectRoot 'tools\doctor.py') @args
if ($LASTEXITCODE -ne 0) { throw 'Diagnostics found missing components.' }




