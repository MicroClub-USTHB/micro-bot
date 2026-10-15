<#
.SYNOPSIS
    Stage 1 of the simulation runbook: prove the environment can run the benchmark before anything
    is started (specs/vps-simulation, docs/simulation/4gb-vps.md).

.DESCRIPTION
    Checks, in order: the Docker CLI and a running daemon, a Compose version that understands
    !override, the presence of .env, that both overlay combinations parse, and what the simulation
    will actually get (port, limits, volumes). Prints the resolved configuration with every secret
    value redacted, because `docker compose config` prints keys in clear text. Read-only: it starts
    and stops nothing.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/simulation/Test-SimPreflight.ps1
#>
[CmdletBinding()]
param()

. "$PSScriptRoot/SimCommon.ps1"

$repo = Get-SimRepoRoot
Push-Location $repo
try {
    Write-SimStep 'docker CLI and daemon'
    Test-SimDocker
    $dockerInfo = docker info --format '{{.NCPU}} cpus / {{.MemTotal}} bytes / {{.OSType}}'
    Write-Host "  $dockerInfo"

    Write-SimStep 'compose version'
    $composeVersion = (docker compose version --short).Trim()
    Write-Host "  docker compose $composeVersion"
    if ([version]($composeVersion -replace '^v', '') -lt [version]'2.24.0') {
        Write-Error 'compose < 2.24 does not support !override, which the overlays need'; exit 3
    }

    Write-SimStep '.env'
    if (-not (Test-Path -LiteralPath (Get-SimEnvFile))) {
        Write-Error ".env is missing; copy .env.example to .env and fill in the tokens"; exit 4
    }
    $envValues = Get-SimEnvValues
    Write-Host "  $(@(Get-Content -LiteralPath (Get-SimEnvFile)).Count) lines, $(@($envValues.Keys).Count) settings"
    $hasKey = $envValues.ContainsKey('OPENROUTER_API_KEY') -and $envValues['OPENROUTER_API_KEY'].Length -gt 0
    Write-Host "  scenario: $(if ($hasKey) { 'S1 (provider key set, generation will be called)' } else { 'S0 (no key, retrieval only)' })"

    Write-SimStep 'resolved simulation configuration (secrets redacted)'
    $simArgs = @($SimBaseFiles)
    $resolved = docker compose @simArgs config 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host (Protect-SimText ($resolved -join [Environment]::NewLine))
        Write-Error 'the simulation configuration does not parse'; exit 5
    }
    $text = Protect-SimText ($resolved -join [Environment]::NewLine)
    foreach ($pattern in @('name: mc-rag-sim', 'host_ip: 127.0.0.1', 'published: "8001"',
            'mem_limit', 'memswap_limit', 'cpus', 'mc-rag-sim_app_data', 'mc-rag-sim_qdrant_storage')) {
        $hit = if ($text -match [regex]::Escape($pattern)) { 'ok  ' } else { 'MISS' }
        Write-Host "  $hit $pattern"
    }
    $capacity = docker compose @($SimBaseFiles + @('-f', $SimCapacityFile)) config 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host '  ok   capacity overlay parses (Profile B)'
    } else {
        Write-Host (Protect-SimText ($capacity -join [Environment]::NewLine))
        Write-Error 'the capacity overlay does not parse'; exit 5
    }

    Write-SimStep 'simulation-only settings must not be in .env'
    $simEnv = @{
        QDRANT_COLLECTION          = 'mcclub_sim'
        INGEST_EMBED_THREADS       = '2'
        RETRIEVAL_RERANK_THREADS   = '2'
        RETRIEVAL_RERANK_WORKERS   = '1'
        INGEST_MAX_CONCURRENCY     = '1'
    }
    foreach ($key in $simEnv.Keys) {
        if ($envValues.ContainsKey($key)) {
            $verdict = if ($envValues[$key] -ne $simEnv[$key]) { 'ok   (dev value kept)' } else { 'CONFLICT' }
            Write-Host "  $verdict $key is set in .env as $($envValues[$key])"
            if ($verdict -eq 'CONFLICT') { exit 6 }
        } else {
            Write-Host "  ok   $key only in the overlay"
        }
    }

    Write-SimStep 'VM limits'
    $wslConfig = Join-Path $env:USERPROFILE '.wslconfig'
    if (Test-Path -LiteralPath $wslConfig) {
        Write-Host "  .wslconfig: $((Get-Content -LiteralPath $wslConfig | Where-Object { $_ -match '^\s*(memory|processors|swap)=' }) -join ', ')"
    } else {
        Write-Host '  no .wslconfig: the VM still has Docker Desktop defaults (see Set-SimVmLimits.ps1)' -ForegroundColor Yellow
    }

    Write-SimStep 'the normal deployment is not part of this project'
    # a label filter, not a label template: PowerShell strips the quotes inside {{.Label "..."}}
    $others = @(docker ps --filter "label=com.docker.compose.project!=$SimProject" --format '{{.Names}} ({{.Status}})' 2>$null)
    if ($others) { $others | ForEach-Object { "  running: $_" } } else { '  no containers from another project are running' }
    Write-Host ''
    Write-Host 'preflight passed' -ForegroundColor Green
} finally {
    Pop-Location
}
