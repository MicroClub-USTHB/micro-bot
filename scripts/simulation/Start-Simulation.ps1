<#
.SYNOPSIS
    Stage 3 of the runbook: start the isolated simulation and wait until it answers
    (specs/vps-simulation, docs/simulation/4gb-vps.md).

.DESCRIPTION
    Brings up the mc-rag-sim project (optionally building the image first), polls the health
    endpoint until the models are loaded, then prints what the containers were actually given:
    memory limit, memory+swap limit, CPU quota, OOM flag and restart count. Health reporting
    "degraded" is expected in scenario S0 (no provider key) and does not mean retrieval is broken.

.PARAMETER Build
    Build the application image first. Do this before applying the 4 GB VM ceiling: the build
    installs the dependencies and downloads ~700 MB of models.

.PARAMETER Profile
    B adds docker-compose.simulation.capacity.yml (chat rate limit 1000/60 s), for capacity runs.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/simulation/Start-Simulation.ps1 -Build
#>
[CmdletBinding()]
param(
    [switch]$Build,
    [ValidateSet('A', 'B')]
    [string]$Profile = 'A',
    [int]$HealthTimeoutSeconds = 0
)

. "$PSScriptRoot/SimCommon.ps1"

# the default cannot reference $SimHealthTimeoutSeconds: param defaults bind before the dot-source
if ($HealthTimeoutSeconds -le 0) { $HealthTimeoutSeconds = $SimHealthTimeoutSeconds }

Test-SimDocker
$repo = Get-SimRepoRoot
Push-Location $repo
try {
    $compose = @($SimBaseFiles)
    if ($Profile -eq 'B') { $compose += @('-f', $SimCapacityFile) }

    if ($Build) {
        Write-SimStep 'building the application image (ceiling still at host values)'
        Write-Host '  models: bge-m3 568 MB, mmarco 119 MB, BM25 stopwords, ara/fra/eng tessdata'
        docker compose @compose build app
        if ($LASTEXITCODE -ne 0) { Write-Error 'the image build failed'; exit 1 }
    }

    Write-SimStep 'pulling qdrant'
    docker compose @compose pull --quiet qdrant
    if ($LASTEXITCODE -ne 0) { Write-Error 'could not pull the qdrant image'; exit 1 }

    Write-SimStep 'starting the simulation (project mc-rag-sim)'
    docker compose @compose up -d
    if ($LASTEXITCODE -ne 0) { Write-Error 'docker compose up failed'; exit 1 }

    Write-SimStep "waiting for $SimAppUrl/api/v1/health (models load on the first call)"
    $deadline = (Get-Date).AddSeconds($HealthTimeoutSeconds)
    $health = $null
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 3
        try {
            $health = Invoke-RestMethod -Uri "$SimAppUrl/api/v1/health" -TimeoutSec 5
            break
        } catch {
            $health = $null
        }
    }
    if (-not $health) {
        Write-Host (docker compose @compose logs --tail 40 app)
        Write-Error "the app did not answer within $HealthTimeoutSeconds s"; exit 5
    }
    Write-Host "  health: $($health | ConvertTo-Json -Compress)"
    if ($health.status -ne 'ok') {
        Write-Host '  degraded is expected in S0: health also probes the generation provider.' -ForegroundColor Yellow
    }

    Write-SimStep 'what the containers were given'
    docker compose @compose ps --format 'table {{.Name}}\t{{.Status}}\t{{.Ports}}'
    foreach ($name in (Get-SimContainerState).Keys) {
        $state = (Get-SimContainerState)[$name]
        $memGb = [math]::Round([int64]$state.mem_limit / 1GB, 2)
        $swapGb = [math]::Round([int64]$state.swap_limit / 1GB, 2)
        $cpus = [math]::Round([int64]$state.nano_cpus / 1e9, 2)
        Write-Host ("  {0,-28} mem {1} GB, mem+swap {2} GB, cpus {3}, oom_killed {4}, restarts {5}" -f
            $name, $memGb, $swapGb, $cpus, $state.oom_killed, $state.restarts)
        if ([int64]$state.swap_limit -ne [int64]$state.mem_limit) {
            Write-Host '    warning: memswap_limit differs from mem_limit, the container may swap' -ForegroundColor Yellow
        }
    }

    Write-Host ''
    Write-Host "simulation up on $SimAppUrl (profile $Profile)" -ForegroundColor Green
} finally {
    Pop-Location
}
