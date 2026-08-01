[CmdletBinding()]
param(
    [switch]$Plan,
    [switch]$SkipModels,
    [switch]$SkipShortcut
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimeRoot = if ($env:DUBBING_STUDIO_RUNTIME) { $env:DUBBING_STUDIO_RUNTIME } else { Join-Path $ProjectRoot 'runtime' }
$LogRoot = Join-Path $ProjectRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$LogFile = Join-Path $LogRoot ('setup-{0}.log' -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Start-Transcript -LiteralPath $LogFile | Out-Null

function Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
}

function Find-Python310 {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310\python.exe'),
        (Join-Path $env:ProgramFiles 'Python310\python.exe'),
        'C:\Python310\python.exe'
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -ErrorAction SilentlyContinue) { return $candidate }
    }
    return $null
}

function Install-WingetPackage([string]$Id, [string]$Name) {
    if ($Plan) { Write-Host "[plan] install $Name ($Id)"; return }
    Step "Installing $Name"
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        & winget install --exact --id $Id --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) { throw "winget could not install $Name ($Id)." }
    }
    else {
        $Bootstrap = Join-Path $RuntimeRoot 'bootstrap'
        New-Item -ItemType Directory -Force -Path $Bootstrap | Out-Null
        if ($Id -eq 'Python.Python.3.10') {
            $Installer = Join-Path $Bootstrap 'python-3.10.11-amd64.exe'
            Invoke-WebRequest 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe' -OutFile $Installer
            $process = Start-Process $Installer -ArgumentList '/quiet InstallAllUsers=0 PrependPath=1 Include_test=0' -Wait -PassThru
        }
        elseif ($Id -eq 'Git.Git') {
            $Installer = Join-Path $Bootstrap 'Git-2.51.0-64-bit.exe'
            Invoke-WebRequest 'https://github.com/git-for-windows/git/releases/download/v2.51.0.windows.1/Git-2.51.0-64-bit.exe' -OutFile $Installer
            $process = Start-Process $Installer -ArgumentList '/VERYSILENT /NORESTART /NOCANCEL /SP-' -Wait -PassThru
        }
        elseif ($Id -eq 'Gyan.FFmpeg') {
            $Archive = Join-Path $Bootstrap 'ffmpeg-release-essentials.zip'
            $Destination = Join-Path $RuntimeRoot 'tools\ffmpeg'
            Invoke-WebRequest 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile $Archive
            Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force
            $ffmpeg = Get-ChildItem -LiteralPath $Destination -Recurse -Filter ffmpeg.exe | Select-Object -First 1
            if (-not $ffmpeg) { throw 'FFmpeg was downloaded, but ffmpeg.exe is missing from the archive.' }
            $env:Path = "$($ffmpeg.Directory.FullName);$env:Path"
            return
        }
        else { throw "No fallback installer is available for $Id." }
        if ($process.ExitCode -ne 0) { throw "The $Name installer exited with code $($process.ExitCode)." }
    }
    Refresh-Path
}

function Ensure-Venv([string]$Path, [string]$BasePython) {
    $python = Join-Path $Path 'Scripts\python.exe'
    $healthy = $false
    if (Test-Path -LiteralPath $python) {
        if ($Plan) { $healthy = $true }
        else {
            & $python -c 'import sys; assert sys.version_info[:2] == (3, 10)' 2>$null
            $healthy = $LASTEXITCODE -eq 0
        }
    }
    if (-not $healthy) {
        if ($Plan) { Write-Host "[plan] create environment $Path"; return $python }
        if (Test-Path -LiteralPath $Path) {
            $backup = "$Path.broken-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
            Move-Item -LiteralPath $Path -Destination $backup
            Write-Host "The broken environment was preserved at: $backup" -ForegroundColor Yellow
        }
        & $BasePython -m venv $Path
    }
    return $python
}

function Pip([string]$Python, [string[]]$Arguments) {
    if ($Plan) { Write-Host "[plan] $Python -m pip $($Arguments -join ' ')"; return }
    & $Python -m pip @Arguments
    if ($LASTEXITCODE -ne 0) { throw "pip failed: $($Arguments -join ' ')" }
}

function Ensure-Repo([string]$Url, [string]$Path, [string]$Commit) {
    if (-not (Test-Path -LiteralPath (Join-Path $Path '.git'))) {
        if ($Plan) { Write-Host "[plan] git clone $Url -> $Path @ $Commit"; return }
        & git clone $Url $Path
        if ($LASTEXITCODE -ne 0) { throw "Could not clone $Url" }
    }
    if (-not $Plan) {
        & git -C $Path fetch --quiet origin $Commit
        & git -C $Path checkout --quiet --detach $Commit
        if ($LASTEXITCODE -ne 0) { throw "Could not pin $Url to $Commit" }
    }
}

try {
    Step 'Checking the system'
    if ($env:OS -ne 'Windows_NT') { throw 'Automatic setup currently supports Windows 10/11 only.' }
    New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Install-WingetPackage 'Git.Git' 'Git' }
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { Install-WingetPackage 'Gyan.FFmpeg' 'FFmpeg' }
    $BasePython = Find-Python310
    if (-not $BasePython) {
        Install-WingetPackage 'Python.Python.3.10' 'Python 3.10'
        $BasePython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310\python.exe'
    }
    if (-not $Plan -and -not (Test-Path -LiteralPath $BasePython)) { throw 'Python 3.10 was installed but could not be found. Restart setup.bat.' }
    $portableFfmpeg = Get-ChildItem -LiteralPath (Join-Path $RuntimeRoot 'tools\ffmpeg') -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($portableFfmpeg) { $env:Path = "$($portableFfmpeg.Directory.FullName);$env:Path" }
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        throw 'No NVIDIA driver was found. Install the latest NVIDIA Studio Driver and run setup.bat again.'
    }
    $drive = Get-PSDrive -Name ([IO.Path]::GetPathRoot($RuntimeRoot).TrimEnd(':\'))
    $freeGb = [math]::Round($drive.Free / 1GB, 1)
    if ($freeGb -lt 45 -and -not $SkipModels) { throw "Only $freeGb GB is free on this drive. Setup requires at least 45 GB." }

    $AppPython = Ensure-Venv (Join-Path $ProjectRoot '.venv') $BasePython
    Step 'Installing the web application'
    Pip $AppPython @('install', '--upgrade', 'pip', 'wheel')
    Pip $AppPython @('install', '-r', (Join-Path $ProjectRoot 'requirements.txt'))

    $AsrRoot = Join-Path $RuntimeRoot 'asr'
    New-Item -ItemType Directory -Force -Path $AsrRoot | Out-Null
    $AsrPython = Ensure-Venv (Join-Path $AsrRoot '.venv') $BasePython
    Step 'Installing speech recognition and audio separation'
    Pip $AsrPython @('install', '--upgrade', 'pip', 'wheel')
    Pip $AsrPython @('install', 'torch==2.7.1', 'torchaudio==2.7.1', '--index-url', 'https://download.pytorch.org/whl/cu128')
    Pip $AsrPython @('install', '-r', (Join-Path $ProjectRoot 'requirements\asr.txt'))

    $QwenRoot = Join-Path $RuntimeRoot 'qwen3-tts'
    Step 'Installing Qwen3-TTS'
    Ensure-Repo 'https://github.com/QwenLM/Qwen3-TTS.git' $QwenRoot '022e286b98fbec7e1e916cb940cdf532cd9f488e'
    $QwenPython = Ensure-Venv (Join-Path $QwenRoot '.venv') $BasePython
    Pip $QwenPython @('install', '--upgrade', 'pip', 'wheel')
    Pip $QwenPython @('install', 'torch==2.7.1', 'torchaudio==2.7.1', '--index-url', 'https://download.pytorch.org/whl/cu128')
    Pip $QwenPython @('install', '-e', $QwenRoot)

    $HyRoot = Join-Path $RuntimeRoot 'hymt'
    New-Item -ItemType Directory -Force -Path $HyRoot | Out-Null
    $HyPython = Ensure-Venv (Join-Path $HyRoot '.venv') $BasePython
    Step 'Installing the local Hy-MT2 translator'
    Pip $HyPython @('install', '--upgrade', 'pip', 'wheel')
    Pip $HyPython @('install', 'torch==2.7.1', '--index-url', 'https://download.pytorch.org/whl/cu128')
    Pip $HyPython @('install', '-r', (Join-Path $ProjectRoot 'requirements\hymt.txt'))

    $SeedRoot = Join-Path $RuntimeRoot 'seed-vc'
    Step 'Installing Seed-VC and voice quality tools'
    Ensure-Repo 'https://github.com/Plachtaa/seed-vc.git' $SeedRoot '51383efd921027683c89e5348211d93ff12ac2a8'
    $SeedPython = Ensure-Venv (Join-Path $SeedRoot '.venv') $BasePython
    Pip $SeedPython @('install', '--upgrade', 'pip', 'wheel')
    Pip $SeedPython @('install', 'torch==2.7.1', 'torchvision==0.22.1', 'torchaudio==2.7.1', '--index-url', 'https://download.pytorch.org/whl/cu128')
    Pip $SeedPython @('install', '-r', (Join-Path $ProjectRoot 'requirements\seedvc-extra.txt'))

    if (-not $SkipModels) {
        Step 'Downloading AI models (about 30 GB; interrupted downloads can resume safely)'
        if (-not $Plan) { & $AppPython (Join-Path $ProjectRoot 'tools\download_models.py') --runtime $RuntimeRoot }
        else { Write-Host '[plan] download pinned Hugging Face model revisions' }
        Step 'Preparing the Demucs model'
        if (-not $Plan) {
            $env:TORCH_HOME = Join-Path $AsrRoot 'models\torch'
            & $AsrPython (Join-Path $ProjectRoot 'tools\prefetch_demucs.py')
        }
    }

    if (-not $Plan) {
        $state = @{ installed_at = (Get-Date).ToString('o'); runtime = $RuntimeRoot; version = '0.1.0' } | ConvertTo-Json
        Set-Content -LiteralPath (Join-Path $RuntimeRoot 'install-state.json') -Value $state -Encoding UTF8
        Step 'Running final diagnostics'
        & (Join-Path $ProjectRoot 'doctor.ps1')
        if (-not $SkipShortcut) { & (Join-Path $ProjectRoot 'create-shortcut.ps1') }
    }
    if ($Plan) { Write-Host "`nSetup plan validated successfully; nothing was downloaded." -ForegroundColor Green }
    else { Write-Host "`nDubbing Studio installed successfully." -ForegroundColor Green }
}
finally {
    try { Stop-Transcript | Out-Null } catch {}
    Write-Host "Log: $LogFile" -ForegroundColor DarkGray
}







