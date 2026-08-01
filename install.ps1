# Compatibility wrapper for the legacy command. The full installer is setup.ps1.
& (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'setup.ps1') @args

