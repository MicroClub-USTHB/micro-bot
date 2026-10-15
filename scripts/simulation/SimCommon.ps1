# Shared helpers for the 4 GB / 2-core simulation scripts (specs/vps-simulation).
# Dot-sourced: . "$PSScriptRoot/SimCommon.ps1". Windows PowerShell 5.1 compatible.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$SimProject = 'mc-rag-sim'
$SimBaseFiles = @('-f', 'docker-compose.yml', '-f', 'docker-compose.simulation.yml')
$SimCapacityFile = 'docker-compose.simulation.capacity.yml'
$SimAppUrl = 'http://127.0.0.1:8001'
$SimHealthTimeoutSeconds = 300
$SimSecrets = @('OPENROUTER_API_KEY', 'MEMBER_TOKEN', 'ADMIN_API_KEY', 'EVAL_JUDGE_API_KEY')

function Get-SimRepoRoot {
    Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
}

function Get-SimEnvFile {
    Join-Path (Get-SimRepoRoot) '.env'
}

function Get-SimEnvValues {
    <# KEY=VALUE pairs of .env, comments stripped. Values are never printed. #>
    $path = Get-SimEnvFile
    if (-not (Test-Path -LiteralPath $path)) { return @{} }
    $values = @{}
    foreach ($line in Get-Content -LiteralPath $path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#') -or -not $trimmed.Contains('=')) { continue }
        $parts = $trimmed.Split('=', 2)
        $values[$parts[0].Trim()] = $parts[1].Trim().Trim('"')
    }
    return $values
}

function Protect-SimText {
    <# Redacts secret values in any text (docker compose config prints them in clear). #>
    param([Parameter(ValueFromPipeline = $true)][string]$Text)
    $out = $Text
    foreach ($pair in (Get-SimEnvValues).GetEnumerator()) {
        if ($SimSecrets -contains $pair.Key -and $pair.Value.Length -gt 3) {
            $out = $out.Replace($pair.Value, '***redacted***')
        }
    }
    return $out
}

function Test-SimDocker {
    <# Throws with an actionable message unless the Docker daemon answers. Exit code 2. #>
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error 'docker is not on PATH. Install Docker Desktop.'; exit 2
    }
    # stderr from a native command is a terminating error under 'Stop', so relax it just here
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & docker info --format '{{.NCPU}}' 2>$null | Out-Null
        $answers = $LASTEXITCODE -eq 0
    } finally {
        $ErrorActionPreference = $previous
    }
    if (-not $answers) {
        Write-Error 'the Docker daemon is not answering. Start Docker Desktop and re-run.'; exit 2
    }
}

function Get-SimComposeArgs {
    <# The -f arguments for the simulation, optionally with the Profile B overlay. #>
    param([switch]$Capacity)
    $args_ = @($SimBaseFiles)
    if ($Capacity) { $args_ += @('-f', $SimCapacityFile) }
    Push-Location (Get-SimRepoRoot)
    return $args_
}

function Invoke-SimCompose {
    <# Runs docker compose for the simulation project from the repo root. #>
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$ComposeArgs)
    Push-Location (Get-SimRepoRoot)
    try {
        & docker compose @ComposeArgs
        return $LASTEXITCODE
    } finally {
        Pop-Location
    }
}

function Get-SimContainerState {
    <# name -> OOMKilled / RestartCount / memory limit, read from the live containers. #>
    Push-Location (Get-SimRepoRoot)
    try {
        $ids = & docker compose --project-name $SimProject ps --quiet
        $state = @{}
        foreach ($id in @($ids)) {
            if (-not $id) { continue }
            $name = (& docker inspect --format '{{.Name}}' $id).TrimStart('/')
            $state[$name] = [ordered]@{
                oom_killed    = (& docker inspect --format '{{.State.OOMKilled}}' $id).Trim()
                restarts      = (& docker inspect --format '{{.RestartCount}}' $id).Trim()
                mem_limit     = (& docker inspect --format '{{.HostConfig.Memory}}' $id).Trim()
                swap_limit    = (& docker inspect --format '{{.HostConfig.MemorySwap}}' $id).Trim()
                nano_cpus     = (& docker inspect --format '{{.HostConfig.NanoCpus}}' $id).Trim()
            }
        }
        return $state
    } finally {
        Pop-Location
    }
}

function Write-SimStep {
    param([string]$Text)
    Write-Host ''
    Write-Host "== $Text" -ForegroundColor Cyan
}
