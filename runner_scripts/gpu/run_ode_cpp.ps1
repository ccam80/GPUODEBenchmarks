# Windows MPGOS runner: enters the Visual Studio developer environment and calls nvcc.
param(
    # Exact aliases keep -a, -g and -s unambiguous under prefix matching.
    [Alias('a')]
    [ValidateSet('performance', 'work-precision', 'states', 'warm')]
    [string]$Analysis = 'performance',
    # A single value is a sweep ceiling (8, 32, ... <= n); a comma list runs exactly those Ns.
    [Alias('n')]
    [string]$Nmax = '16777216',
    [Alias('g')]
    [string]$Algorithm = 'all',
    [Alias('s')]
    [string]$Problem = 'all'
)

$ErrorActionPreference = 'Stop'

if ($Nmax -notmatch '^\d+(,\d+)*$') {
    Write-Host "-n/--nmax must be a positive integer or a comma list of them, got '$Nmax'"
    exit 1
}
if ($Nmax.Contains(',')) {
    $NValues = @($Nmax.Split(',') | ForEach-Object { [long]$_ })
} else {
    $NValues = @()
    $next = [long]8
    while ($next -le [long]$Nmax) {
        $NValues += $next
        $next = $next * 4
    }
}

# MPGOS solvers: RK4 (classical-rk4, fixed) and RKCK45 (cash-karp-54, adaptive).
$Solvers = switch ($Algorithm) {
    'all' { @('RK4', 'RKCK45') }
    'classical-rk4' { @('RK4') }
    'cash-karp-54' { @('RKCK45') }
    default {
        Write-Host "MPGOS does not support algorithm '$Algorithm'; skipping."
        exit 0
    }
}

# Load modules eagerly so the first-launch cubin load stays out of timed regions.
$env:CUDA_MODULE_LOADING = 'EAGER'

Push-Location (Join-Path $PSScriptRoot '..\..')

$Problems = @(& python runner_scripts\mpgos_problems.py $Problem)
if ($Problems.Count -eq 0) {
    Write-Host "MPGOS runs none of the requested problems; skipping."
    Pop-Location
    exit 0
}

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

# Built binaries are cached per source hash, machine and build constants.
$DatasetKey = (& powershell -ExecutionPolicy Bypass -File "runner_scripts\bench_key.ps1").Trim()
$SourceFiles = @((Resolve-Path "GPU_ODE_MPGOS\Bench.cu").Path,
    (Resolve-Path "GPU_ODE_MPGOS\makefile").Path) +
    @(Get-ChildItem "GPU_ODE_MPGOS\problems", "GPU_ODE_MPGOS\SourceCodes" -Recurse -File |
      Sort-Object FullName | ForEach-Object { $_.FullName })
$Hasher = [System.Security.Cryptography.SHA256]::Create()
$SrcBytes = [byte[]]@()
foreach ($f in $SourceFiles) { $SrcBytes += [System.IO.File]::ReadAllBytes($f) }
$SrcHash = ([System.BitConverter]::ToString($Hasher.ComputeHash($SrcBytes)) -replace '-', '').Substring(0, 12)
$CacheDir = "GPU_ODE_MPGOS\build_cache\$DatasetKey"

# Mirrors GPU_ODE_MPGOS/Makefile; reuses the cached binary unless -Fresh.
function Build-Project {
    param([string]$ProblemName, [string]$Solver, [long]$Nt, [long]$Sd = 0,
          [switch]$Fresh)
    $SdTag = if ($Sd -gt 0) { "_SD$Sd" } else { "" }
    $CachedExe = "$CacheDir\Bench_${ProblemName}_${Solver}_NT$Nt${SdTag}_$SrcHash.exe"
    if (-not $Fresh -and (Test-Path $CachedExe)) {
        Copy-Item $CachedExe "GPU_ODE_MPGOS\Bench.exe" -Force
        Write-Host "Cached build: $(Split-Path $CachedExe -Leaf)"
        return
    }
    if (Test-Path "GPU_ODE_MPGOS\Bench.exe") {
        Remove-Item "GPU_ODE_MPGOS\Bench.exe" -Force
    }
    $SdDefine = if ($Sd -gt 0) { "-DPROBLEM_SD=$Sd" } else { $null }
    nvcc -o GPU_ODE_MPGOS\Bench.exe GPU_ODE_MPGOS\Bench.cu `
        -I"GPU_ODE_MPGOS\SourceCodes" -I"GPU_ODE_MPGOS" `
        "-DPROBLEM_HEADER=\`"problems/$ProblemName.cuh\`"" `
        "-DSOLVER_CHOICE=$Solver" "-DNT_VALUE=$Nt" $SdDefine `
        -O3 -std=c++17 --ptxas-options=-v --gpu-architecture=native `
        -lineinfo -maxrregcount=128
    if ($LASTEXITCODE -ne 0) {
        Write-Error "nvcc build failed with exit code $LASTEXITCODE"
    }
    if (-not $Fresh) {
        New-Item -ItemType Directory -Force $CacheDir | Out-Null
        Copy-Item "GPU_ODE_MPGOS\Bench.exe" $CachedExe -Force
    }
}

function Invoke-Point {
    param([string]$ProblemName, [string]$Solver, [long]$Nt, [switch]$Wp)
    Build-Project -ProblemName $ProblemName -Solver $Solver -Nt $Nt
    if ($Wp) {
        & "GPU_ODE_MPGOS\Bench.exe" wp
    } else {
        & "GPU_ODE_MPGOS\Bench.exe"
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Bench.exe ($ProblemName, $Solver) failed with exit code $LASTEXITCODE"
    }
}

Enter-VsEnvironment

# Every missing target builds through parallel nvcc straight into the cache.
function Invoke-WarmBuilds {
    param([object[]]$Targets)
    $jobsMax = 8
    if ($env:BENCH_WARM_JOBS) { $jobsMax = [int]$env:BENCH_WARM_JOBS }
    New-Item -ItemType Directory -Force $CacheDir | Out-Null
    $builds = @()
    foreach ($t in $Targets) {
        $p, $s, $nt, $sd = $t
        $sdTag = if ($sd -gt 0) { "_SD$sd" } else { "" }
        $exe = "$CacheDir\Bench_${p}_${s}_NT$nt${sdTag}_$SrcHash.exe"
        if (Test-Path $exe) { continue }
        while (@($builds | Where-Object { -not $_.Proc.HasExited }).Count -ge $jobsMax) {
            Start-Sleep -Seconds 2
        }
        Write-Host "building $(Split-Path $exe -Leaf)"
        $nvccArgs = @('-o', $exe, 'GPU_ODE_MPGOS\Bench.cu',
            '-IGPU_ODE_MPGOS\SourceCodes', '-IGPU_ODE_MPGOS',
            "-DPROBLEM_HEADER=\`"problems/$p.cuh\`"", "-DSOLVER_CHOICE=$s",
            "-DNT_VALUE=$nt")
        if ($sd -gt 0) { $nvccArgs += "-DPROBLEM_SD=$sd" }
        $nvccArgs += @('-O3', '-std=c++17', '--ptxas-options=-v',
            '--gpu-architecture=native', '-lineinfo', '-maxrregcount=128')
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

function Get-NtTargets {
    $targets = @()
    $nts = @($NValues + [long]131072 | Sort-Object -Unique)
    foreach ($p in $Problems) {
        foreach ($s in $Solvers) {
            foreach ($nt in $nts) { $targets += , @($p, $s, $nt, [long]0) }
        }
    }
    return $targets
}

if ($Analysis -eq 'warm') {
    Invoke-WarmBuilds -Targets @(Get-NtTargets)
    Pop-Location
    exit 0
}

# All binaries compile in parallel before anything is timed.
if ($Analysis -eq 'performance') {
    Invoke-WarmBuilds -Targets @(Get-NtTargets)
}

if ($Analysis -eq 'states') {
    $StatesN = [long]131072
    $Grid = (& python runner_scripts\problems.py --states-grid).Trim() -split ' '
    Remove-Item "data\CPP\$DatasetKey\lorenz96\MPGOS_states_*.txt" -Force -ErrorAction SilentlyContinue
    foreach ($solver in $Solvers) {
        foreach ($n in $Grid) {
            Write-Host "lorenz96 states = $n ($solver, N=$StatesN)"
            $Watch = [System.Diagnostics.Stopwatch]::StartNew()
            Build-Project -ProblemName lorenz96 -Solver $solver -Nt $StatesN -Sd ([long]$n) -Fresh
            $BuildS = [string]::Format([System.Globalization.CultureInfo]::InvariantCulture,
                "{0:F3}", $Watch.Elapsed.TotalSeconds)
            & "GPU_ODE_MPGOS\Bench.exe" states $BuildS
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Bench.exe (lorenz96 states=$n, $solver) failed with exit code $LASTEXITCODE"
            }
        }
    }
    Pop-Location
    exit 0
}

foreach ($problemName in $Problems) {
    if ($Analysis -eq 'work-precision') {
        foreach ($solver in $Solvers) {
            Invoke-Point -ProblemName $problemName -Solver $solver -Nt 131072 -Wp
        }
        continue
    }
    # NT is a compile-time constant, so every point is a rebuild.
    foreach ($solver in $Solvers) {
        foreach ($a in $NValues) {
            Write-Host "No. of trajectories = $a ($problemName, $solver)"
            Invoke-Point -ProblemName $problemName -Solver $solver -Nt $a
        }
    }
}

Pop-Location
