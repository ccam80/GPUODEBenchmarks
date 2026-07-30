# Windows counterpart of run_full_dataset.sh: drive the complete benchmark
# dataset in one set-and-forget run. Same stages, flags, failure policy, logs
# and outputs; see run_full_dataset.sh for the full description.
#
# Usage (run_full_dataset.bat forwards to this script):
#   run_full_dataset.bat                      # everything, nmax = 2^30
#   run_full_dataset.bat -n 16777216          # smaller ceiling
#   run_full_dataset.bat --skip-ne            # drop a stage
#   run_full_dataset.bat --resume-from jax    # restart at a framework
#   run_full_dataset.bat --only overlap       # a single stage
#   run_full_dataset.bat --lock-clocks 1470,6801   # override the clock target
#   run_full_dataset.bat --no-lock-clocks     # sample clocks but do not pin
#   run_full_dataset.bat --clock-tolerance 30 # widen the drift threshold (MHz)
#
# Exit code: 0 if every stage and framework succeeded, 1 if any did not.
# A non-zero exit is expected when frameworks OOM at high N; read the summary
# table to see how far each one got. Clock drift during a timed stage also
# fails the run.

Set-Location $PSScriptRoot

$NMax = [long][math]::Pow(2, 30)
$DoPerf = $true
$DoWp = $true
$DoNe = $true
$DoOverlap = $true
$DoPlots = $true
$OverlapProfile = 'full'
$OverlapNMax = ''
$Cooldown = 15
$ResumeFrom = ''
$AllowUnknownGpu = $false
$LockClocks = $true
$ClockTarget = ''        # "SM[,MEM]"; empty means use the per-GPU table
$PlotAll = $false

$Languages = @('julia', 'cpp', 'pytorch', 'jax', 'cubie', 'cubie_mlir', 'myokit_cuda')

. .\runner_scripts\clock_guard.ps1
. .\runner_scripts\bench_key.ps1

function Show-Usage {
    param([int]$Code = 0)
    Get-Content $PSCommandPath -TotalCount 18 |
        ForEach-Object { $_ -replace '^# ?', '' }
    exit $Code
}

# Under --only plots, redraw everything from disk; otherwise plots track DoPerf/DoWp.
function Set-OnlyStage {
    param([string]$Stage)
    $script:DoPerf = $false; $script:DoWp = $false; $script:DoNe = $false
    $script:DoOverlap = $false; $script:DoPlots = $false
    switch ($Stage) {
        { $_ -in 'perf', 'performance' } { $script:DoPerf = $true; $script:DoPlots = $true }
        { $_ -in 'wp', 'work-precision' } { $script:DoWp = $true; $script:DoPlots = $true }
        { $_ -in 'ne', 'numerical-equivalence' } { $script:DoNe = $true }
        'overlap' { $script:DoOverlap = $true }
        'plots' { $script:DoPlots = $true; $script:PlotAll = $true }
        default { Write-Host "Unknown stage '$Stage' (perf|wp|ne|overlap|plots)"; exit 1 }
    }
}

function Get-RequiredValue {
    param([object[]]$Arguments, [int]$Index, [string]$Flag)
    if ($Index + 1 -ge $Arguments.Count) { Write-Host "$Flag requires a value"; exit 1 }
    return [string]$Arguments[$Index + 1]
}

for ($i = 0; $i -lt $args.Count; $i++) {
    switch -Regex ([string]$args[$i]) {
        '^(-n|--nmax)$' { $NMax = [long](Get-RequiredValue $args $i $args[$i]); $i++ }
        '^--overlap-nmax$' { $OverlapNMax = Get-RequiredValue $args $i $args[$i]; $i++ }
        '^--overlap-profile$' { $OverlapProfile = Get-RequiredValue $args $i $args[$i]; $i++ }
        '^--resume-from$' { $ResumeFrom = (Get-RequiredValue $args $i $args[$i]) -replace '-', '_'; $i++ }
        '^--cooldown$' { $Cooldown = [int](Get-RequiredValue $args $i $args[$i]); $i++ }
        '^--only$' { Set-OnlyStage (Get-RequiredValue $args $i $args[$i]); $i++ }
        '^--skip-perf$' { $DoPerf = $false }
        '^--skip-wp$' { $DoWp = $false }
        '^--skip-ne$' { $DoNe = $false }
        '^--skip-overlap$' { $DoOverlap = $false }
        '^--skip-plots$' { $DoPlots = $false }
        '^--allow-unknown-gpu$' { $AllowUnknownGpu = $true }
        '^--lock-clocks$' { $ClockTarget = Get-RequiredValue $args $i $args[$i]; $LockClocks = $true; $i++ }
        '^--no-lock-clocks$' { $LockClocks = $false }
        '^--clock-tolerance$' { $script:ClockTolMhz = [int](Get-RequiredValue $args $i $args[$i]); $i++ }
        '^(-h|--help)$' { Show-Usage 0 }
        default { Write-Host "Unknown option $($args[$i])"; Show-Usage 1 }
    }
}

# The overlap suite is slower per point, so its ceiling can be capped separately.
if (-not $OverlapNMax) { $OverlapNMax = $NMax }

$DatasetKey = Get-DatasetKey

# Refuse to key hours of output to an unidentifiable GPU.
if ($DatasetKey -match '_unknown-gpu$' -and -not $AllowUnknownGpu) {
    Write-Host "X Could not identify the GPU - dataset key would be '$DatasetKey'."
    Write-Host "  Fix the driver, or pass --allow-unknown-gpu to run anyway."
    exit 1
}

$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$LogDir = "logs\${DatasetKey}_$Stamp"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Results = Join-Path $LogDir 'summary.tsv'
New-Item -ItemType File -Force -Path $Results | Out-Null

# Pin before any GPU work starts; unpin on every exit path via the finally below.
$ClockStatus = 'off'
if ($LockClocks) {
    if (Set-ClockTargets $DatasetKey $ClockTarget) {
        if (Lock-GpuClocks) {
            $ClockStatus = "locked SM=$script:ClockSm"
            if ($script:ClockMem) { $ClockStatus += " MEM=$script:ClockMem" }
        } else {
            $ClockStatus = "unlocked (not admin) - target was SM=$script:ClockSm"
        }
    } else {
        $ClockStatus = 'unlocked (no target configured)'
    }
}

# data\<dir>\<prefix>_times_*.txt per framework, for the progress report.
function Get-DataDirFor {
    param([string]$Lang)
    switch ($Lang) {
        'julia' { return 'Julia' }
        'cpp' { return 'CPP' }
        default { return $Lang.ToUpperInvariant() }
    }
}
function Get-DataPrefixFor {
    param([string]$Lang)
    switch ($Lang) {
        'julia' { return 'Julia' } 'cpp' { return 'MPGOS' } 'jax' { return 'Jax' }
        'pytorch' { return 'Torch' } 'cubie' { return 'Cubie' }
        'cubie_mlir' { return 'Cubie_mlir' } 'myokit_cuda' { return 'Myokit_cuda' }
    }
}

# Largest N actually recorded, so a truncated sweep is visible in the summary.
function Get-MaxNReached {
    param([string]$Lang)
    $dir = Join-Path 'data' (Get-DataDirFor $Lang)
    $prefix = Get-DataPrefixFor $Lang
    $best = [long]0
    if (-not (Test-Path $dir)) { return 0 }
    $files = Get-ChildItem -Path $dir -Filter "${prefix}_times_*_$DatasetKey.txt" -ErrorAction SilentlyContinue
    foreach ($file in $files) {
        foreach ($line in Get-Content $file.FullName -ErrorAction SilentlyContinue) {
            $tok = ($line.Trim() -split '\s+')[0] -replace '\..*$', ''
            $n = [long]0
            if ([long]::TryParse($tok, [ref]$n) -and $n -gt $best) { $best = $n }
        }
    }
    return $best
}

function Add-Record {
    param([string]$Stage, [string]$Status, [string]$Detail, [string]$Code)
    Add-Content -Path $Results -Value (($Stage, $Status, $Detail, $Code) -join "`t")
}

function Write-Rule { Write-Host ('=' * 60) }

# ClockCritical: drift fails the step. ClockCheck=$false skips non-GPU steps.
$ClockCritical = $true
$ClockCheck = $true
$ClockFailures = 0
$StepLabel = ''

# Run one labelled step under cmd, tee'd to its own log, never aborting the run.
function Invoke-Step {
    param([string]$Label, [string]$LogFile, [string]$CommandLine)
    Write-Rule
    Write-Host "[$((Get-Date).ToUniversalTime().ToString('HH:mm:ss'))Z] $Label"
    Write-Rule
    $start = Get-Date
    $cstart = Get-ClockStamp
    # Out-Host keeps the tee'd lines out of the function's return value.
    cmd /c "$CommandLine 2>&1" | Tee-Object -FilePath (Join-Path $LogDir $LogFile) | Out-Host
    $status = $LASTEXITCODE
    $cend = Get-ClockStamp -End
    $elapsed = [int]((Get-Date) - $start).TotalSeconds
    if ($status -eq 0) {
        Write-Host "OK $Label  (${elapsed}s)"
    } else {
        Write-Host "X $Label failed with exit $status  (${elapsed}s) - continuing"
    }
    # Check this step's slice of the whole-run clock log.
    if ($script:ClockCheck) {
        $stepName = $Label
        if ($script:StepLabel) { $stepName = $script:StepLabel }
        if (-not (Test-ClockSlice $cstart $cend $stepName $script:ClockCritical)) {
            $script:ClockFailures++
        }
    }
    return $status
}

Write-Host "Dataset key : $DatasetKey"
Write-Host "nmax        : $NMax"
Write-Host "Overlap     : profile=$OverlapProfile nmax=$OverlapNMax"
Write-Host "Log dir     : $LogDir"
Write-Host "Stages      : perf=$DoPerf wp=$DoWp ne=$DoNe overlap=$DoOverlap plots=$DoPlots"
Write-Host "Clocks      : $ClockStatus"
if ($ResumeFrom) { Write-Host "Resume from : $ResumeFrom" }
Write-Host ''

# Provenance for the dataset this run produces.
$gitRev = (& git rev-parse HEAD 2>$null)
if (-not $gitRev) { $gitRev = 'unknown' }
$gitDirty = 'no'
if (& git status --porcelain 2>$null) { $gitDirty = 'yes' }
$manifest = @(
    "dataset_key=$DatasetKey"
    "started_utc=$((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))"
    "nmax=$NMax"
    "overlap_profile=$OverlapProfile"
    "overlap_nmax=$OverlapNMax"
    "git_rev=$gitRev"
    "git_dirty=$gitDirty"
    "host=$env:COMPUTERNAME $([System.Environment]::OSVersion.VersionString)"
    "clocks=$ClockStatus"
    "clock_target_sm=$(if ($script:ClockSm) { $script:ClockSm } else { 'none' })"
    "clock_target_mem=$(if ($script:ClockMem) { $script:ClockMem } else { 'none' })"
    "clock_tolerance_mhz=$script:ClockTolMhz"
)
$gpuInfo = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
foreach ($line in $gpuInfo) { $manifest += "gpu=$line" }
Set-Content -Path (Join-Path $LogDir 'run_manifest.txt') -Value $manifest

$ExitCode = 0
try {
    Start-ClockMonitor (Join-Path $LogDir 'clocks.csv') | Out-Null

    $skipping = [bool]$ResumeFrom

    # ------------------------------------------------------------ performance
    if ($DoPerf) {
        foreach ($lang in $Languages) {
            if ($skipping) {
                if ($lang -eq $ResumeFrom) { $skipping = $false } else {
                    Write-Host "-- skipping $lang (before --resume-from $ResumeFrom)"
                    Add-Record "perf:$lang" 'SKIPPED' '-' '-'
                    continue
                }
            }
            $ClockCritical = $true; $StepLabel = "perf:$lang"
            $status = Invoke-Step "Performance sweep: $lang (nmax=$NMax)" "perf_$lang.log" `
                ".\run_benchmark.bat -l $lang -d gpu -m ode -n $NMax"
            $reached = Get-MaxNReached $lang
            if ($status -eq 0) {
                Add-Record "perf:$lang" 'OK' "maxN=$reached" "$status"
            } elseif ($reached -gt 0) {
                # Partial data survived: the sweep died partway (typically OOM).
                Add-Record "perf:$lang" 'PARTIAL' "maxN=$reached" "$status"
                Write-Host "  -> kept results up to N=$reached; higher N left empty"
            } else {
                Add-Record "perf:$lang" 'FAILED' 'no data' "$status"
            }
            Start-Sleep -Seconds $Cooldown
        }
    }

    # --------------------------------------------------------- work-precision
    if ($DoWp) {
        # The golden reference is generated once up front; every wp sweep needs it.
        if (-not (Test-Path 'data\numerical\golden_lorenz_32768.csv')) {
            # Reference generation is scored on accuracy, not speed.
            $ClockCritical = $false; $StepLabel = 'wp:golden'
            $status = Invoke-Step 'Golden reference for work-precision' 'wp_golden.log' `
                'julia -t auto --project=. runner_scripts\golden\generate_golden.jl'
            if ($status -eq 0) {
                Add-Record 'wp:golden' 'OK' '-' "$status"
            } else {
                Add-Record 'wp:golden' 'FAILED' 'wp sweeps cannot score' "$status"
                Write-Host '  -> work-precision sweeps will fail without the golden reference'
            }
        } else {
            Add-Record 'wp:golden' 'OK' 'already present' '0'
        }

        foreach ($lang in $Languages) {
            $ClockCritical = $true; $StepLabel = "wp:$lang"
            $status = Invoke-Step "Work-precision sweep: $lang" "wp_$lang.log" `
                ".\run_benchmark.bat -l $lang -d gpu -m ode -w"
            if ($status -eq 0) { Add-Record "wp:$lang" 'OK' '-' "$status" }
            else { Add-Record "wp:$lang" 'FAILED' '-' "$status" }
            Start-Sleep -Seconds $Cooldown
        }
    }

    # -------------------------------------------------- numerical equivalence
    if ($DoNe) {
        # Equivalence is a correctness check; its clock does not have to be stable.
        $ClockCritical = $false; $StepLabel = 'ne'
        $status = Invoke-Step 'Numerical-equivalence suite (all)' 'numerical_equivalence.log' `
            '.\run_numerical_equivalence.bat all'
        # Exit 2 means a mismatching algorithm, not an infrastructure failure.
        switch ($status) {
            0 { Add-Record 'ne' 'OK' 'all equivalent' "$status" }
            2 { Add-Record 'ne' 'MISMATCH' 'see numerical_equivalence_*.md' "$status" }
            default { Add-Record 'ne' 'FAILED' '-' "$status" }
        }
    }

    # ----------------------------------------------- cubie vs DiffEqGPU overlap
    if ($DoOverlap) {
        $py = 'GPU_ODE_CUBIE\venv\Scripts\python.exe'
        if (-not (Test-Path $py)) { $py = 'python' }
        $ClockCritical = $true; $StepLabel = 'overlap'
        $status = Invoke-Step "Cubie vs DiffEqGPU overlap ($OverlapProfile, nmax=$OverlapNMax)" `
            'cubie_julia_overlap.log' `
            "$py run_cubie_julia_overlap.py --profile $OverlapProfile --phase all --nmax $OverlapNMax"
        # A non-zero exit means at least one worker died, not that all did.
        if ($status -eq 0) { Add-Record 'overlap' 'OK' '-' "$status" }
        else { Add-Record 'overlap' 'PARTIAL' 'a worker failed; see manifest.json' "$status" }
    }

    # ------------------------------------------------------ plots and reports
    if ($DoPlots) {
        # Plotting and reporting do no timed GPU work.
        $ClockCritical = $false; $ClockCheck = $false; $StepLabel = ''
        if ($DoPerf -or $PlotAll) {
            $status = Invoke-Step 'Timing comparison plot' 'plot_ode_comp.log' `
                'julia --project=. runner_scripts\plot\plot_ode_comp.jl'
            if ($status -eq 0) { Add-Record 'plot:timing' 'OK' '-' "$status" }
            else { Add-Record 'plot:timing' 'FAILED' '-' "$status" }
        }

        if ($DoWp -or $PlotAll) {
            $status = Invoke-Step 'Work-precision plot' 'plot_ode_wp.log' `
                'julia --project=. runner_scripts\plot\plot_ode_wp.jl'
            if ($status -eq 0) { Add-Record 'plot:wp' 'OK' '-' "$status" }
            else { Add-Record 'plot:wp' 'FAILED' '-' "$status" }
        }

        # Pairwise numerical comparison needs >=2 keyed datasets.
        $py = 'GPU_ODE_CUBIE\venv\Scripts\python.exe'
        if (Test-Path $py) {
            $status = Invoke-Step 'Pairwise numerical comparison' 'compare_numerical.log' `
                "$py compare_numerical_results.py"
            if ($status -eq 0) { Add-Record 'compare:pairwise' 'OK' '-' "$status" }
            else { Add-Record 'compare:pairwise' 'SKIPPED/FAILED' 'needs >=2 datasets' "$status" }
        } else {
            Add-Record 'compare:pairwise' 'SKIPPED' 'cubie venv missing' '-'
        }
    }

    # ------------------------------------------------------------------ summary
    Write-Host ''
    Write-Rule
    Write-Host "RUN SUMMARY  ($DatasetKey)"
    Write-Rule
    Write-Host ('{0,-26} {1,-16} {2}' -f 'STAGE', 'STATUS', 'DETAIL')
    Write-Host ('{0,-26} {1,-16} {2}' -f '-----', '------', '------')
    $failures = 0
    $partials = 0
    foreach ($row in Get-Content $Results) {
        $c = $row -split "`t"
        if ($c.Count -lt 3) { continue }
        Write-Host ('{0,-26} {1,-16} {2}' -f $c[0], $c[1], $c[2])
        if ($c[1] -eq 'FAILED') { $failures++ }
        if ($c[1] -eq 'PARTIAL') { $partials++ }
    }
    Write-Rule

    Stop-ClockMonitor
    if (Show-ClockReport) { Write-Rule }

    Add-Content -Path (Join-Path $LogDir 'run_manifest.txt') -Value `
        "finished_utc=$((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))"
    Write-Host "Logs: $LogDir"
    Write-Host 'Data: .\data    Plots: .\plots'
    Write-Host "Clocks: $ClockStatus  (1 Hz log in $LogDir\clocks.csv)"
    if ($partials -gt 0) {
        Write-Host "$partials stage(s) partial - expected when frameworks OOM at high N."
    }
    if ($ClockFailures -gt 0) {
        Write-Host "X $ClockFailures timed stage(s) drifted. Lower the lock in $script:ClockConf"
        Write-Host '  and re-run them with --resume-from.'
    }
    if ($failures -gt 0) {
        Write-Host "$failures stage(s) failed outright."
        $ExitCode = 1
    }
    if ($ClockFailures -gt 0) { $ExitCode = 1 }
}
finally {
    Stop-ClockMonitor
    Reset-GpuClocks
}
exit $ExitCode
