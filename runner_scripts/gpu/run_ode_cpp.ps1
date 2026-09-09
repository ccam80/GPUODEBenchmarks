# run_ode_cpp.ps1 --trials <jsonl> [--floor]: builds the binaries the file needs in the VS developer shell, runs its solve trials through Bench.exe; exits the watchdog code when a trial never returned.
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = 'Stop'

$Trials = ''
$Floor = $false
for ($i = 0; $i -lt $Arguments.Count; $i++) {
    switch ($Arguments[$i]) {
        '--trials' { $i++; $Trials = $Arguments[$i] }
        '--floor' { $Floor = $true }
        default { Write-Host "run_ode_cpp.ps1: unknown argument '$($Arguments[$i])'"; exit 1 }
    }
}
if (-not $Trials) { Write-Host 'run_ode_cpp.ps1 --trials <jsonl> [--floor]'; exit 1 }
$Trials = (Resolve-Path $Trials).Path

# Load modules eagerly so the first-launch cubin load stays out of timed regions.
$env:CUDA_MODULE_LOADING = 'EAGER'

Push-Location (Join-Path $PSScriptRoot '..\..')

# The suite interpreter runs the store.
$Python = 'python'
if (Test-Path 'GPU_ODE_CUBIE\venv\Scripts\python.exe') {
    $Python = (Resolve-Path 'GPU_ODE_CUBIE\venv\Scripts\python.exe').Path
}

function Invoke-Listing {
    param([string[]]$ScriptArgs)
    $lines = @(& $Python runner_scripts\mpgos_trials.py @ScriptArgs)
    if ($LASTEXITCODE -ne 0) { Write-Host "mpgos_trials.py $($ScriptArgs -join ' ') failed"; Pop-Location; exit 1 }
    return $lines
}

$Context = @{}
foreach ($line in (Invoke-Listing @('context'))) {
    $name, $value = $line -split '=', 2
    $Context[$name] = $value
}
$DatasetKey = $Context['key']
$SrcHash = $Context['source_hash']
$PackageVersion = $Context['package_version']
$SuiteRev = $Context['suite_rev']
$WatchdogExit = [int]$Context['watchdog_exit']
$CacheDir = "GPU_ODE_MPGOS\build_cache\$DatasetKey"

function Enter-VsEnvironment {
    if (Get-Command cl -ErrorAction SilentlyContinue) {
        return
    }
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        Write-Error "vswhere.exe not found; install Visual Studio with the C++ workload."
    }
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsPath) {
        Write-Error "No Visual Studio installation with C++ tools found."
    }
    Import-Module (Join-Path $vsPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64' | Out-Null
}

Enter-VsEnvironment

# Binaries are cached per source hash, machine and build constants.
function Get-ExePath {
    param([string]$ProblemName, [string]$Solver, [string]$Nt, [string]$Sd, [string]$Precision)
    $sdTag = if ($Sd -ne '-') { "_SD$Sd" } else { '' }
    return "$CacheDir\Bench_${ProblemName}_${Solver}_NT${Nt}${sdTag}_${Precision}_$SrcHash.exe"
}

function Get-NvccArgs {
    param([string]$Exe, [string]$ProblemName, [string]$Solver, [string]$Nt, [string]$Sd, [string]$Precision)
    $type = if ($Precision -eq 'float64') { 'double' } else { 'float' }
    $nvccArgs = @('-o', $Exe, 'GPU_ODE_MPGOS\Bench.cu',
        '-IGPU_ODE_MPGOS\SourceCodes', '-IGPU_ODE_MPGOS',
        "-DPROBLEM_HEADER=\`"problems/$ProblemName.cuh\`"", "-DSOLVER_CHOICE=$Solver",
        "-DNT_VALUE=$Nt", "-DPRECISION_TYPE=$type")
    if ($Sd -ne '-') { $nvccArgs += "-DPROBLEM_SD=$Sd" }
    $nvccArgs += @('-O3', '-std=c++17', '--ptxas-options=-v',
        '--gpu-architecture=native', '-lineinfo', '-maxrregcount=128')
    return $nvccArgs
}

# Warm targets build in parallel into the cache.
function Invoke-WarmBuilds {
    param([object[]]$Targets)
    $jobsMax = 8
    New-Item -ItemType Directory -Force $CacheDir | Out-Null
    $builds = @()
    foreach ($t in $Targets) {
        $exe = Get-ExePath $t.problem $t.solver $t.nt $t.sd $t.precision
        if (Test-Path $exe) { continue }
        while (@($builds | Where-Object { -not $_.Proc.HasExited }).Count -ge $jobsMax) {
            Start-Sleep -Seconds 2
        }
        Write-Host "building $(Split-Path $exe -Leaf)"
        $nvccArgs = Get-NvccArgs $exe $t.problem $t.solver $t.nt $t.sd $t.precision
        $proc = Start-Process nvcc -ArgumentList $nvccArgs -NoNewWindow -PassThru `
            -RedirectStandardOutput "$exe.out" -RedirectStandardError "$exe.err"
        # Caching the handle keeps ExitCode readable after the process ends.
        $null = $proc.Handle
        $builds += @{ Proc = $proc; Exe = $exe }
    }
    foreach ($b in $builds) { $b.Proc.WaitForExit() }
    $failed = @($builds | Where-Object { $_.Proc.ExitCode -ne 0 })
    foreach ($b in $failed) {
        Remove-Item $b.Exe -Force -ErrorAction SilentlyContinue
        Write-Host "FAILED $(Split-Path $b.Exe -Leaf)"
        Get-Content "$($b.Exe).out", "$($b.Exe).err" -ErrorAction SilentlyContinue |
            Select-Object -Last 6 | ForEach-Object { Write-Host "  $_" }
    }
    Remove-Item "$CacheDir\*.out", "$CacheDir\*.err" -Force -ErrorAction SilentlyContinue
    Write-Host "MPGOS builds ready ($($builds.Count - $failed.Count) built, $($failed.Count) failed)."
}

# A cold target builds afresh; its wall time is build_s.
function Invoke-ColdBuild {
    param([object]$Target)
    $exe = Get-ExePath $Target.problem $Target.solver $Target.nt $Target.sd $Target.precision
    New-Item -ItemType Directory -Force $CacheDir | Out-Null
    Remove-Item $exe -Force -ErrorAction SilentlyContinue
    Write-Host "cold build $(Split-Path $exe -Leaf)"
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $nvccArgs = Get-NvccArgs $exe $Target.problem $Target.solver $Target.nt $Target.sd $Target.precision
    # Build output goes to files; only the seconds return.
    $proc = Start-Process nvcc -ArgumentList $nvccArgs -NoNewWindow -PassThru -Wait `
        -RedirectStandardOutput "$exe.out" -RedirectStandardError "$exe.err"
    $seconds = [string]::Format([System.Globalization.CultureInfo]::InvariantCulture,
        '{0:F3}', $watch.Elapsed.TotalSeconds)
    if ($proc.ExitCode -ne 0) {
        Remove-Item $exe -Force -ErrorAction SilentlyContinue
        Write-Host "FAILED $(Split-Path $exe -Leaf)"
        Get-Content "$exe.out", "$exe.err" -ErrorAction SilentlyContinue |
            Select-Object -Last 6 | ForEach-Object { Write-Host "  $_" }
    }
    Remove-Item "$exe.out", "$exe.err" -Force -ErrorAction SilentlyContinue
    return $seconds
}

function ConvertFrom-Row {
    param([string]$Line, [string[]]$Columns)
    $cells = $Line -split "`t", $Columns.Count
    $row = @{}
    for ($c = 0; $c -lt $Columns.Count; $c++) { $row[$Columns[$c]] = $cells[$c] }
    return $row
}

$BuildColumns = @('problem', 'solver', 'nt', 'sd', 'precision', 'cold', 'leg')
$PointColumns = @('trial_id', 'leg', 'ordinal', 'problem', 'solver', 'nt', 'sd', 'precision', 'transfers', 'finals', 'reason')

$Builds = @(Invoke-Listing @('builds', $Trials) | ForEach-Object { ConvertFrom-Row $_ $BuildColumns })
$Points = @(Invoke-Listing @('points', $Trials) | ForEach-Object { ConvertFrom-Row $_ $PointColumns })

# build_s per leg, from the cold builds.
$BuildSeconds = @{}
Invoke-WarmBuilds -Targets @($Builds | Where-Object { $_.cold -ne 'true' })
foreach ($b in @($Builds | Where-Object { $_.cold -eq 'true' })) {
    $BuildSeconds[$b.leg] = Invoke-ColdBuild -Target $b
}

# NaN rows for a point the script could not run.
function Add-NanRows {
    param([object]$Point, [string]$Transfers, [string]$Reason)
    $nanArgs = @('nan', $Trials, $Point.trial_id, $DatasetKey, $Transfers, $Reason)
    if ($Floor) { $nanArgs += '--floor' }
    if ($BuildSeconds.ContainsKey($Point.leg)) { $nanArgs += @('--build-s', $BuildSeconds[$Point.leg]) }
    & $Python runner_scripts\mpgos_trials.py @nanArgs
    if ($LASTEXITCODE -ne 0) { Write-Host "mpgos_trials.py nan failed for $($Point.trial_id)"; Pop-Location; exit 1 }
    Write-Host "cpp $($Point.problem) $($Point.leg) ordinal $($Point.ordinal) ${Transfers}: $Reason"
}

# (leg|transfers) pairs abandoned after a timeout or oom outcome.
$Abandoned = @{}
$Outcome = "$Trials.outcome"

foreach ($p in $Points) {
    $transfers = @($p.transfers -split ',' | Where-Object { $_ -and -not $Abandoned.ContainsKey("$($p.leg)|$_") })
    if ($transfers.Count -eq 0) { continue }
    $transfersText = $transfers -join ','
    if ($p.reason) {
        Add-NanRows -Point $p -Transfers $transfersText -Reason $p.reason
        continue
    }
    $exe = Get-ExePath $p.problem $p.solver $p.nt $p.sd $p.precision
    if (-not (Test-Path $exe)) {
        Add-NanRows -Point $p -Transfers $transfersText -Reason "error: BuildError: nvcc failed for $(Split-Path $exe -Leaf)"
        continue
    }
    Remove-Item $Outcome -Force -ErrorAction SilentlyContinue
    $benchArgs = @('--trials', $Trials, '--trial', $p.trial_id, '--key', $DatasetKey,
        '--transfers', $transfersText, '--python', $Python, '--package-version', $PackageVersion,
        '--suite-rev', $SuiteRev, '--outcome', $Outcome)
    if ($Floor) { $benchArgs += '--floor' }
    if ($BuildSeconds.ContainsKey($p.leg)) { $benchArgs += @('--build-s', $BuildSeconds[$p.leg]) }
    Write-Host "cpp $($p.leg) ordinal $($p.ordinal) n=$($p.nt) ($transfersText)"
    & $exe @benchArgs
    $code = $LASTEXITCODE
    if ($code -eq $WatchdogExit) {
        Pop-Location
        exit $WatchdogExit
    }
    $done = @{}
    if (Test-Path $Outcome) {
        foreach ($line in Get-Content $Outcome) {
            $which, $result = $line -split ' ', 2
            $done[$which] = $result
            if ($result -eq 'timeout' -or $result -eq 'oom') { $Abandoned["$($p.leg)|$which"] = $true }
        }
    }
    if ($code -ne 0) {
        $missing = @($transfers | Where-Object { -not $done.ContainsKey($_) })
        Write-Host "FAILED $($p.leg) ordinal $($p.ordinal): Bench.exe exit $code"
        if ($missing.Count -gt 0) {
            Add-NanRows -Point $p -Transfers ($missing -join ',') -Reason "error: ProcessError: Bench.exe exit $code"
        }
    }
}

Pop-Location
exit 0
