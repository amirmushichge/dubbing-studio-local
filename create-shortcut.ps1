$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktop 'Dubbing Studio.lnk'
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = Join-Path $projectRoot 'start.bat'
$shortcut.WorkingDirectory = $projectRoot
$shortcut.Description = 'Локальный перевод и дубляж видео'
$shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,168"
$shortcut.Save()
Write-Host "Shortcut created: $shortcutPath" -ForegroundColor Green



