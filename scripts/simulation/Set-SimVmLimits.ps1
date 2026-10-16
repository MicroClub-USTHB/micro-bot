<#
.SYNOPSIS
    Show, apply or restore the WSL 2 VM limits that make Docker Desktop's Linux VM behave like a
    4 GB / 2-core VPS (specs/vps-simulation, docs/simulation/4gb-vps.md).

.DESCRIPTION
    Without %UserProfile%\.wslconfig the Docker Desktop VM silently takes half the host's memory
    and every core: measured here, 7.72 GB and 12 CPUs on a 16 GB / 12-core laptop. This script
    never overwrites that file blindly. Apply backs it up first (or records that there was none),
    merges only memory / processors / swap into the existing [wsl2] section, and prints - without
    running - the two commands needed to make WSL pick the change up.

.PARAMETER Mode
    Show     print the file, its [wsl2] keys and the limits Docker actually has now
    Apply    back up, then merge memory=4GB, processors=2, swap=0
    Restore  put the backup back (or remove the keys if the file did not exist)

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/simulation/Set-SimVmLimits.ps1 -Mode Show
#>
[CmdletBinding()]
param(
    [ValidateSet('Show', 'Apply', 'Restore')]
    [string]$Mode = 'Show',
    [string]$Memory = '4GB',
    [int]$Processors = 2,
    [string]$Swap = '0'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$WslConfig = Join-Path $env:USERPROFILE '.wslconfig'
$BackupGlob = '.wslconfig.sim-backup-*'

function Get-WslSection {
    <# The raw lines of the [wsl2] section, or nothing when the file has none. #>
    param([string[]]$Lines)
    $inSection = $false
    foreach ($line in $Lines) {
        $trimmed = $line.Trim()
        if ($trimmed -match '^\[(.+)\]$') { $inSection = $Matches[1] -eq 'wsl2'; continue }
        if ($inSection -and $trimmed -and -not $trimmed.StartsWith('#') -and $trimmed.Contains('=')) {
            $trimmed
        }
    }
}

function Merge-WslKeys {
    <# Sets the three keys inside [wsl2], keeps every other line, section and key. #>
    param([string[]]$Lines, [hashtable]$Wanted)
    $out = New-Object System.Collections.Generic.List[string]
    $inSection = $false
    $written = @{}
    foreach ($line in $Lines) {
        $trimmed = $line.Trim()
        if ($trimmed -match '^\[(.+)\]$') {
            # close the previous wsl2 section once its keys are all written
            if ($inSection) {
                foreach ($key in $Wanted.Keys) {
                    if (-not $written.ContainsKey($key)) { $out.Add("$key=$($Wanted[$key])") }
                }
            }
            $inSection = $Matches[1] -eq 'wsl2'
            $out.Add($line)
            continue
        }
        if ($inSection -and $trimmed.Contains('=') -and -not $trimmed.StartsWith('#')) {
            $key = $trimmed.Split('=')[0].Trim()
            if ($Wanted.ContainsKey($key)) {
                $out.Add("$key=$($Wanted[$key])")
                $written[$key] = $true
                continue
            }
        }
        $out.Add($line)
    }
    if ($inSection) {
        foreach ($key in $Wanted.Keys) {
            if (-not $written.ContainsKey($key)) { $out.Add("$key=$($Wanted[$key])") }
        }
    } elseif ($written.Count -eq 0) {
        $out.Add('')
        $out.Add('[wsl2]')
        foreach ($key in $Wanted.Keys) { $out.Add("$key=$($Wanted[$key])") }
    }
    return $out.ToArray()
}

function Get-EffectiveLimits {
    # The daemon is usually stopped on purpose here, and with $ErrorActionPreference = 'Stop' a
    # native command's stderr would become a terminating error before Apply could even write.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $info = & docker info --format '{{.NCPU}} cpus, {{.MemTotal}} bytes' 2>$null
        if ($LASTEXITCODE -ne 0) { return 'docker is not answering (normal while Docker Desktop is stopped)' }
        return ($info | Out-String).Trim()
    } finally {
        $ErrorActionPreference = $previous
    }
}

Write-Host "WSL 2 VM limits ($Mode)" -ForegroundColor Cyan
Write-Host "file:   $WslConfig"
Write-Host "docker: $(Get-EffectiveLimits)"
Write-Host ''

switch ($Mode) {
    'Show' {
        if (Test-Path -LiteralPath $WslConfig) {
            Write-Host '[wsl2] keys now set:'
            $keys = Get-WslSection (Get-Content -LiteralPath $WslConfig)
            if ($keys) { $keys | ForEach-Object { "  $_" } } else { '  (none)' }
        } else {
            Write-Host 'no .wslconfig: the VM uses Docker Desktop defaults (half the host RAM, all cores).'
        }
        Write-Host ''
        Write-Host "wanted: memory=$Memory processors=$Processors swap=$Swap"
        Write-Host "expected with those: docker info shows 2 cpus and about 4294000000 bytes"
    }
    'Apply' {
        $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
        $backup = Join-Path $env:USERPROFILE ".wslconfig.sim-backup-$stamp"
        $existed = Test-Path -LiteralPath $WslConfig
        if ($existed) {
            Copy-Item -LiteralPath $WslConfig -Destination $backup -ErrorAction Stop
            if (-not (Test-Path -LiteralPath $backup)) { Write-Error 'backup failed; refusing to write'; exit 1 }
            Write-Host "backed up to: $backup"
        } else {
            Set-Content -LiteralPath $backup -Value '# no .wslconfig existed when the limits were applied' -Encoding UTF8
            Write-Host "no .wslconfig existed; recorded that in $backup"
        }
        $lines = if ($existed) { Get-Content -LiteralPath $WslConfig } else { @() }
        $merged = Merge-WslKeys -Lines $lines -Wanted @{ memory = $Memory; processors = $Processors; swap = $Swap }
        Set-Content -LiteralPath $WslConfig -Value $merged -Encoding UTF8
        Write-Host ''
        Write-Host 'written:'
        Get-Content -LiteralPath $WslConfig | ForEach-Object { "  $_" }
        Write-Host ''
        Write-Host 'Now run these yourself (they stop the running VM):' -ForegroundColor Yellow
        Write-Host '  1. quit Docker Desktop (tray icon -> Quit)'
        Write-Host '  2. wsl --shutdown'
        Write-Host '  3. start Docker Desktop again'
        Write-Host "  4. re-run with -Mode Show and check docker reports 2 cpus / ~4 GB"
    }
    'Restore' {
        $backups = @(Get-ChildItem -LiteralPath $env:USERPROFILE -Filter $BackupGlob -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending)
        if ($backups.Count -eq 0) { Write-Error "no $BackupGlob backup found; nothing to restore"; exit 1 }
        $latest = $backups[0]
        $first = (Get-Content -LiteralPath $latest.FullName | Select-Object -First 1)
        if ($first -match '^# no \.wslconfig existed') {
            if (Test-Path -LiteralPath $WslConfig) { Remove-Item -LiteralPath $WslConfig -Force }
            Write-Host "removed $WslConfig (there was none before)"
        } else {
            Copy-Item -LiteralPath $latest.FullName -Destination $WslConfig -Force
            Write-Host "restored $WslConfig from $($latest.Name)"
        }
        Write-Host 'run "wsl --shutdown", then restart Docker Desktop.'
    }
}
