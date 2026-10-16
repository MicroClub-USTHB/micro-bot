<#
.SYNOPSIS
    Stage 5 / 6 of the runbook: run one labelled load stage against the simulation while sampling
    what the containers actually use (specs/vps-simulation, docs/simulation/4gb-vps.md).

.DESCRIPTION
    Runs the existing e2e load suite over HTTP from Windows, so the load generator stays outside
    the constrained VM, and samples `docker stats` every 2 s into eval/results/simulation/. It
    reports the harness's throughput and percentiles, the peak memory of each container, and any
    OOM kill or restart. MEMBER_TOKEN and ADMIN_API_KEY are read from .env into this process only;
    they are never printed.

.PARAMETER Scenario
    S0 (retrieval only, no provider key) or S1 (real generation through OpenRouter, spends quota).

.PARAMETER Profile
    A keeps the shipped 20/60 s chat rate limit; B expects the capacity overlay (1000/60 s) and
    fails if the running app does not have it.

.PARAMETER RequestsPerMinute
    Maximum offered rate (eval --requests-per-minute). The achieved rate is bounded by latency and
    by how fast the server can answer.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/simulation/Invoke-SimBenchmark.ps1 `
        -Profile B -Concurrency 4 -Duration 120 -RequestsPerMinute 240 -Label sim-b4-240rpm
#>
[CmdletBinding()]
param(
    [ValidateSet('S0', 'S1')]
    [string]$Scenario = 'S0',
    [ValidateSet('A', 'B')]
    [string]$Profile = 'A',
    [int]$Concurrency = 2,
    [int]$Duration = 60,
    [int]$RequestsPerMinute = 120,
    [string]$Label = 'stage',
    [string]$Split = 'dev',
    [int]$Limit = 40,
    [switch]$Upload,
    [int]$SampleSeconds = 2
)

. "$PSScriptRoot/SimCommon.ps1"

Test-SimDocker
$repo = Get-SimRepoRoot
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$outDir = Join-Path $repo 'eval\results\simulation'
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$csv = Join-Path $outDir "$stamp-$Label-stats.csv"

# the harness reads the tokens from the environment (HttpDriver); keep them out of the output
$envValues = Get-SimEnvValues
foreach ($key in @('MEMBER_TOKEN', 'ADMIN_API_KEY')) {
    if (-not $envValues.ContainsKey($key) -or -not $envValues[$key]) {
        Write-Error "$key is not set in .env (the harness needs it for member and admin requests)"; exit 4
    }
    Set-Item -Path "env:$key" -Value $envValues[$key]
}

Push-Location $repo
try {
    Write-SimStep 'preconditions'
    try {
        $health = Invoke-RestMethod -Uri "$SimAppUrl/api/v1/health" -TimeoutSec 5
        Write-Host "  health: $($health | ConvertTo-Json -Compress)"
    } catch {
        Write-Error "the simulation is not answering on $SimAppUrl (run Start-Simulation.ps1 first)"; exit 1
    }
    $state = Get-SimContainerState
    if (-not $state.Count) { Write-Error 'no simulation containers are running'; exit 1 }
    $appEnv = docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' ($state.Keys | Where-Object { $_ -like '*app*' } | Select-Object -First 1)
    $limitInApp = ($appEnv | Select-String -Pattern '^CHAT_RATE_LIMIT_REQUESTS=(.*)$').Matches.Groups[1].Value
    if ($Profile -eq 'B' -and $limitInApp -ne '1000') {
        Write-Error "Profile B needs the capacity overlay: the app has CHAT_RATE_LIMIT_REQUESTS=$limitInApp"
        Write-Error 'restart with: Start-Simulation.ps1 -Profile B'
        exit 7
    }
    if ($Profile -eq 'A') { Write-Host "  profile A: the app's own limit is CHAT_RATE_LIMIT_REQUESTS=$limitInApp" }
    $hasKey = $envValues.ContainsKey('OPENROUTER_API_KEY') -and $envValues['OPENROUTER_API_KEY'].Length -gt 0
    if ($Scenario -eq 'S0' -and $hasKey) {
        Write-Host '  note: .env has a provider key, so the app will call OpenRouter (costs quota)' -ForegroundColor Yellow
    }
    if ($Scenario -eq 'S1' -and -not $hasKey) {
        Write-Error 'S1 needs OPENROUTER_API_KEY in .env (uncomment it) and an app restart'
    }
    Write-Host "  offering up to $RequestsPerMinute requests/minute with $Concurrency concurrent users for $Duration s"

    Write-SimStep "sampling docker stats every $SampleSeconds s -> $csv"
    "timestamp,name,memory_mb,memory_percent,cpu_percent" | Set-Content -LiteralPath $csv -Encoding UTF8
    $sampler = Start-Job -ScriptBlock {
        param($StopFile, $Out, $Every)
        while (-not (Test-Path -LiteralPath $StopFile)) {
            $now = (Get-Date).ToUniversalTime().ToString('o')
            foreach ($line in (& docker stats --no-stream --format '{{.Name}}|{{.MemUsage}}|{{.CPUPerc}}' 2>$null)) {
                $parts = $line -split '\|'
                if ($parts.Count -lt 3) { continue }
                $used = ($parts[1] -split '/')[0].Trim()
                $mb = if ($used -match '^([0-9.]+)\s*([KMG]i?B|B)$') {
                    $number = [double]$Matches[1]
                    switch -Regex ($Matches[2]) {
                        '^(G|Gi)B$' { [math]::Round($number * 1024, 1) }
                        '^(M|Mi)B$' { [math]::Round($number, 1) }
                        '^(K|Ki)B$' { [math]::Round($number / 1024, 1) }
                        default     { [math]::Round($number / 1MB, 1) }
                    }
                } else { 0 }
                "$now,$($parts[0]),$mb,$($parts[1]),$($parts[2])" | Add-Content -LiteralPath $Out
            }
            Start-Sleep -Seconds $Every
        }
    } -ArgumentList (Join-Path $outDir "$stamp.stop"), $csv, $SampleSeconds

    Write-SimStep "load: $Label"
    $evalArgs = @(
        'run', 'python', '-m', 'eval.e2e.run',
        '--target', 'http', '--base-url', $SimAppUrl,
        '--suite', 'load',
        '--concurrency', $Concurrency,
        '--duration', $Duration,
        '--requests-per-minute', $RequestsPerMinute,
        '--split', $Split,
        '--label', $Label
    )
    if ($Limit -gt 0) { $evalArgs += @('--limit', $Limit) }
    if ($Upload) { $evalArgs += '--upload' }
    & uv @evalArgs
    $evalExit = $LASTEXITCODE

    $stopFile = Join-Path $outDir "$stamp.stop"
    Set-Content -LiteralPath $stopFile -Value 'stop' -Encoding UTF8
    Wait-Job $sampler -Timeout 15 | Out-Null
    Remove-Job $sampler -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stopFile -Force -ErrorAction SilentlyContinue

    Write-SimStep 'peak memory seen during the run (docker stats, server side)'
    $rows = Import-Csv -LiteralPath $csv | Where-Object { $_.timestamp }
    foreach ($group in ($rows | Group-Object name)) {
        $peak = ($group.Group | ForEach-Object { [double]$_.memory_mb } | Measure-Object -Maximum).Maximum
        $cpu = ($group.Group | ForEach-Object { ($_.cpu_percent -replace '%', '') } |
            Where-Object { $_ -match '^[0-9.]+$' } | ForEach-Object { [double]$_ } |
            Measure-Object -Maximum).Maximum
        Write-Host ("  {0,-30} peak {1} MB, peak cpu {2}%" -f $group.Name, $peak, [math]::Round($cpu, 1))
    }

    Write-SimStep 'container events'
    foreach ($name in (Get-SimContainerState).Keys) {
        $s = (Get-SimContainerState)[$name]
        Write-Host ("  {0,-30} oom_killed {1}, restarts {2}" -f $name, $s.oom_killed, $s.restarts)
    }

    Write-Host ''
    Write-Host "stats: $csv" -ForegroundColor Green
    Write-Host "eval exit code: $evalExit (0 = thresholds met, 1 = a threshold failed, 2 = the run crashed)"
} finally {
    Pop-Location
}
