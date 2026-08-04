# Drive the complete benchmark dataset in one run; flags match run_full_dataset.sh.
#
# Usage (run_full_dataset.bat forwards to this script):
#   run_full_dataset.bat                           # every analysis, every package
#   run_full_dataset.bat -n 33554432               # larger ceiling
#   run_full_dataset.bat -n 8388608,134217728      # exact trajectory counts only
#   run_full_dataset.bat -p cpp                    # one package
#   run_full_dataset.bat -p cubie,julia            # several packages
#   run_full_dataset.bat -a overlap                # one analysis
#   run_full_dataset.bat -a performance,numerical  # several analyses
#   run_full_dataset.bat -g euler,tsit5            # several algorithms
#   run_full_dataset.bat --resume-from jax         # restart at a package
#   run_full_dataset.bat --lock-clocks 1470,6801   # override the clock target
#   run_full_dataset.bat --no-lock-clocks          # sample clocks but do not pin
#   run_full_dataset.bat --clock-tolerance 30      # widen the drift threshold (MHz)
#
#   -p, --package   all (default) | comma list of julia | cpp | pytorch | jax | cubie | cubie_mlir | myokit_cuda
#   -a, --analysis  all (default) | comma list of performance | work-precision | numerical | overlap | plots
#   -n, --nmax      sweep ceiling (8, 32, ... <= n; default 16777216) or comma list of exact Ns
#   -g, --algorithm all (default) | comma list of euler|classical-rk4|tsit5|cash-karp-54
#
# Exit code: 0 if every analysis and package succeeded, 1 if any did not.
# Clock drift in a timed analysis also fails the run.

Set-Location $PSScriptRoot

$NMax = '16777216'
$DoPerf = $true
$DoWp = $true
$DoNe = $true
$DoOverlap = $true
$DoPlots = $true
$OverlapProfile = 'full'
$Cooldown = 15
$ResumeFrom = ''
$AllowUnknownGpu = $false
$LockClocks = $true
$ClockTarget = ''        # "SM[,MEM]"; empty means use the per-GPU table
$PlotAll = $false
$Package = 'all'
$Algorithm = 'all'

$AllPackages = @('julia', 'cpp', 'pytorch', 'jax', 'cubie', 'cubie_mlir', 'myokit_cuda')

. .\runner_scripts\clock_guard.ps1
. .\runner_scripts\bench_key.ps1

function Show-Usage {
    param([int]$Code = 0)
    Get-Content $PSCommandPath -TotalCount 24 |
        ForEach-Object { $_ -replace '^# ?', '' }
    exit $Code
}

# Selecting plots alone redraws everything on disk; otherwise plots track perf/wp.
function Set-Analyses {
    param([string]$List)
    $script:DoPerf = $false; $script:DoWp = $false; $script:DoNe = $false
    $script:DoOverlap = $false; $script:DoPlots = $false
    foreach ($item in $List.Split(',')) {
        switch ($item.Trim()) {
            'all' {
                $script:DoPerf = $true; $script:DoWp = $true; $script:DoNe = $true
                $script:DoOverlap = $true; $script:DoPlots = $true
            }
            'performance' { $script:DoPerf = $true; $script:DoPlots = $true }
            'work-precision' { $script:DoWp = $true; $script:DoPlots = $true }
            'numerical' { $script:DoNe = $true }
            'overlap' { $script:DoOverlap = $true }
            'plots' { $script:DoPlots = $true; $script:PlotAll = $true }
            default {
                Write-Host "Unknown analysis '$item' (all|performance|work-precision|numerical|overlap|plots)"
                exit 1
            }
        }
    }
}

function Get-RequiredValue {
    param([object[]]$Arguments, [int]$Index, [string]$Flag)
    if ($Index + 1 -ge $Arguments.Count) { Write-Host "$Flag requires a value"; exit 1 }
    # PowerShell turns an unquoted comma list into an array; rejoin it.
    return ([string[]]$Arguments[$Index + 1] -join ',')
}

for ($i = 0; $i -lt $args.Count; $i++) {
    switch -Regex ([string]$args[$i]) {
        '^(-n|--nmax)$' { $NMax = [string](Get-RequiredValue $args $i $args[$i]); $i++ }
        '^(-p|--package)$' { $Package = (Get-RequiredValue $args $i $args[$i]) -replace '-', '_'; $i++ }
        '^(-g|--algorithm)$' { $Algorithm = Get-RequiredValue $args $i $args[$i]; $i++ }
        '^(-a|--analysis)$' { Set-Analyses (Get-RequiredValue $args $i $args[$i]); $i++ }
        '^--profile$' { $OverlapProfile = Get-RequiredValue $args $i $args[$i]; $i++ }
        '^--resume-from$' { $ResumeFrom = (Get-RequiredValue $args $i $args[$i]) -replace '-', '_'; $i++ }
        '^--cooldown$' { $Cooldown = [int](Get-RequiredValue $args $i $args[$i]); $i++ }
        '^--allow-unknown-gpu$' { $AllowUnknownGpu = $true }
        '^--lock-clocks$' { $ClockTarget = Get-RequiredValue $args $i $args[$i]; $LockClocks = $true; $i++ }
        '^--no-lock-clocks$' { $LockClocks = $false }
        '^--clock-tolerance$' { $script:ClockTolMhz = [int](Get-RequiredValue $args $i $args[$i]); $i++ }
        '^(-h|--help)$' { Show-Usage 0 }
        default { Write-Host "Unknown option $($args[$i])"; Show-Usage 1 }
    }
}

# -p accepts "all" or a comma list; each token is validated.
$Languages = @()
$HasAllPackages = $false
foreach ($pkg in ($Package.Split(',') | Where-Object { $_ })) {
    if ($pkg -eq 'all') {
        $HasAllPackages = $true
    } elseif ($AllPackages -contains $pkg) {
        if ($Languages -notcontains $pkg) { $Languages += $pkg }
    } else {
        Write-Host "Unknown package '$pkg' (all|$($AllPackages -join '|'))"
        exit 1
    }
}
if ($HasAllPackages) { $Languages = $AllPackages }
if ($Languages.Count -eq 0) {
    Write-Host "-p/--package requires a value"
    exit 1
}

# ne/overlap take a single -p token: julia+cubie -> all, one -> that one.
$NePackage = ''
$HasJulia = $Languages -contains 'julia'
$HasCubie = $Languages -contains 'cubie'
if ($HasJulia -and $HasCubie) { $NePackage = 'all' }
elseif ($HasJulia) { $NePackage = 'julia' }
elseif ($HasCubie) { $NePackage = 'cubie' }

# --profile: whitelisted before it reaches a command line.
if ($OverlapProfile -notin @('smoke', 'full')) {
    Write-Host "Unknown profile '$OverlapProfile' (smoke|full)"
    exit 1
}

# -g: "all" or a comma list; every token whitelisted.
$AllAlgorithms = @('all', 'euler', 'classical-rk4', 'tsit5', 'cash-karp-54')
$algTokens = @($Algorithm.Split(',') | Where-Object { $_ })
if ($algTokens.Count -eq 0) {
    Write-Host "-g/--algorithm requires a value"
    exit 1
}
foreach ($alg in $algTokens) {
    if ($AllAlgorithms -notcontains $alg) {
        Write-Host "Unknown algorithm '$alg' (all|euler|classical-rk4|tsit5|cash-karp-54)"
        exit 1
    }
}
if ($algTokens -contains 'all') { $Algorithm = 'all' }

# -n: sweep ceiling or comma list of exact counts.
if ($NMax -notmatch '^\d+(,\d+)*$') {
    Write-Host "-n/--nmax must be a positive integer or a comma list of them, got '$NMax'"
    exit 1
}

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
    $files = Get-ChildItem -Path (Join-Path $dir $DatasetKey) -Filter "${prefix}_times_*.txt" -ErrorAction SilentlyContinue
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
Write-Host "Algorithm   : $Algorithm"
Write-Host "Overlap     : profile=$OverlapProfile"
Write-Host "Packages    : $($Languages -join ', ')"
Write-Host "Log dir     : $LogDir"
Write-Host "Analyses    : performance=$DoPerf work-precision=$DoWp numerical=$DoNe overlap=$DoOverlap plots=$DoPlots"
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
    "algorithm=$Algorithm"
    "overlap_profile=$OverlapProfile"
    "packages=$($Languages -join ',')"
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
                ".\run_benchmark.bat -p $lang -d gpu -m ode -a performance -n `"$NMax`" -g `"$Algorithm`""
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
                ".\run_benchmark.bat -p $lang -d gpu -m ode -a work-precision -g `"$Algorithm`""
            if ($status -eq 0) { Add-Record "wp:$lang" 'OK' '-' "$status" }
            else { Add-Record "wp:$lang" 'FAILED' '-' "$status" }
            Start-Sleep -Seconds $Cooldown
        }
    }

    # -------------------------------------------------- numerical equivalence
    if ($DoNe) {
        if (-not $NePackage) {
            Add-Record 'ne' 'SKIPPED' "$Package is not in the ne suite" '-'
        } else {
            # Equivalence is a correctness check; its clock does not have to be stable.
            $ClockCritical = $false; $StepLabel = 'ne'
            $status = Invoke-Step "Numerical equivalence ($NePackage)" 'numerical_equivalence.log' `
                ".\run_numerical_equivalence.bat -p $NePackage"
            # Exit 2 means a mismatching algorithm, not an infrastructure failure.
            switch ($status) {
                0 { Add-Record 'ne' 'OK' 'all equivalent' "$status" }
                2 { Add-Record 'ne' 'MISMATCH' 'see numerical_equivalence_*.md' "$status" }
                default { Add-Record 'ne' 'FAILED' '-' "$status" }
            }
        }
    }

    # ----------------------------------------------- cubie vs DiffEqGPU overlap
    if ($DoOverlap) {
        # Only cubie and julia have an algorithm-for-algorithm mapping.
        if (-not $NePackage) {
            Add-Record 'overlap' 'SKIPPED' "$Package is not in the overlap suite" '-'
        } else {
            $py = 'GPU_ODE_CUBIE\venv\Scripts\python.exe'
            if (-not (Test-Path $py)) { $py = 'python' }
            $ClockCritical = $true; $StepLabel = 'overlap'
            $status = Invoke-Step "Cubie vs DiffEqGPU overlap ($OverlapProfile, n=$NMax)" `
                'cubie_julia_overlap.log' `
                "$py run_cubie_julia_overlap.py --profile $OverlapProfile -a all -p $NePackage -n $NMax"
            # A non-zero exit means at least one worker died, not that all did.
            if ($status -eq 0) { Add-Record 'overlap' 'OK' '-' "$status" }
            else { Add-Record 'overlap' 'PARTIAL' 'a worker failed; see manifest.json' "$status" }
        }
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

        $py = 'GPU_ODE_CUBIE\venv\Scripts\python.exe'
        if (($DoOverlap -or $PlotAll) -and $NePackage) {
            if (Test-Path $py) {
                $status = Invoke-Step 'Overlap plot and report' 'cubie_julia_overlap_analyze.log' `
                    "$py runner_scripts\cubie_julia_overlap\analyze.py"
                if ($status -eq 0) { Add-Record 'plot:overlap' 'OK' '-' "$status" }
                else { Add-Record 'plot:overlap' 'FAILED' '-' "$status" }
            } else {
                Add-Record 'plot:overlap' 'SKIPPED' 'cubie venv missing' '-'
            }
        }

        # Exit 3 is "nothing to compare"; any other non-zero is a real failure.
        if (Test-Path $py) {
            $status = Invoke-Step 'Pairwise numerical comparison' 'compare_numerical.log' `
                "$py compare_numerical_results.py"
            if ($status -eq 0) { Add-Record 'compare:pairwise' 'OK' '-' "$status" }
            elseif ($status -eq 3) { Add-Record 'compare:pairwise' 'SKIPPED' 'needs >=2 keyed datasets' "$status" }
            else { Add-Record 'compare:pairwise' 'FAILED' '-' "$status" }
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
