<#
.SYNOPSIS
    Stage 8 of the runbook: remove the simulation and nothing else
    (specs/vps-simulation, docs/simulation/4gb-vps.md).

.DESCRIPTION
    Runs `docker compose down -v` for the mc-rag-sim project only, which removes its containers, its
    network and its two named volumes (mc-rag-sim_app_data, mc-rag-sim_qdrant_storage). The
    -v flag deletes the simulation's documents and vectors: they are re-created by the next
    Start-Simulation. The normal deployment project is never named, and the script refuses to run
    with any other project name.

.PARAMETER KeepVolumes
    Stop the containers but keep the simulation's data (no -v).

.PARAMETER RemoveImage
    Also delete the project-scoped image mc-rag-sim-app (~1.5 GB, rebuilt by -Build).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/simulation/Stop-Simulation.ps1
#>
[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$KeepVolumes,
    [switch]$RemoveImage
)

. "$PSScriptRoot/SimCommon.ps1"

if ($SimProject -ne 'mc-rag-sim') {
    Write-Error "refusing to run: this script only touches '$SimProject', not '$SimProject'"; exit 1
}

Push-Location (Get-SimRepoRoot)
try {
    Write-SimStep "containers of project $SimProject"
    $containers = @(docker ps --filter "label=com.docker.compose.project=$SimProject" --format '{{.Names}} ({{.Status}})')
    $volumes = @(docker volume ls --filter "label=com.docker.compose.project=$SimProject" --format '{{.Name}}')
    if (-not $containers -and -not $volumes) {
        Write-Host '  nothing to remove'
        exit 0
    }
    $containers | ForEach-Object { "  container $_" }
    $volumes | ForEach-Object { "  volume $_" }

    if ($KeepVolumes) {
        Write-Host ''
        Write-Host '  -KeepVolumes: the volumes above stay; the next run reuses this data'
    } elseif (-not $Force) {
        Write-Host ''
        Write-Host 'This deletes the simulation documents and vectors (-v). The normal' -ForegroundColor Yellow
        Write-Host "deployment's project is NOT touched. Type 'sim' to continue:" -ForegroundColor Yellow
        $answer = Read-Host
        if ($answer -ne 'sim') { Write-Host 'cancelled'; exit 1 }
    } else {
        Write-Host ''
        Write-Host '  -Force: deleting the simulation data without asking' -ForegroundColor Yellow
    }

    $downArgs = @('--project-name', $SimProject, 'down')
    if (-not $KeepVolumes) { $downArgs += '-v' }
    docker compose @SimBaseFiles @downArgs
    if ($LASTEXITCODE -ne 0) { Write-Error 'docker compose down failed'; exit 1 }

    if ($RemoveImage) {
        docker image rm "$SimProject-app" 2>&1 | Out-Null
        Write-Host "  image $SimProject-app removed (rebuild with Start-Simulation.ps1 -Build)"
    }

    Write-Host ''
    Write-Host 'simulation removed; the normal deployment was never named' -ForegroundColor Green
} finally {
    Pop-Location
}
