# Windows MPGOS runner: enters the Visual Studio developer environment and calls nvcc.
param(
    # Exact aliases keep -a, -g and -s unambiguous under prefix matching.
    [Alias('a')]
    [ValidateSet('performance', 'work-precision')]
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

# Mirrors GPU_ODE_MPGOS/Makefile.
function Build-Project {
    param([string]$ProblemName, [string]$Solver, [long]$Nt)
    if (Test-Path "GPU_ODE_MPGOS\Bench.exe") {
        Remove-Item "GPU_ODE_MPGOS\Bench.exe" -Force
    }
    nvcc -o GPU_ODE_MPGOS\Bench.exe GPU_ODE_MPGOS\Bench.cu `
        -I"GPU_ODE_MPGOS\SourceCodes" -I"GPU_ODE_MPGOS" `
        "-DPROBLEM_HEADER=\`"problems/$ProblemName.cuh\`"" `
        "-DSOLVER_CHOICE=$Solver" "-DNT_VALUE=$Nt" `
        -O3 -std=c++17 --ptxas-options=-v --gpu-architecture=native `
        -lineinfo -maxrregcount=128
    if ($LASTEXITCODE -ne 0) {
        Write-Error "nvcc build failed with exit code $LASTEXITCODE"
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

foreach ($problemName in $Problems) {
    if ($Analysis -eq 'work-precision') {
        foreach ($solver in $Solvers) {
            Invoke-Point -ProblemName $problemName -Solver $solver -Nt 131072 -Wp
        }
        continue
    }
    # One solver walks the whole N sweep before the next builds; NT is a
    # compile-time constant, so each point is still a rebuild.
    foreach ($solver in $Solvers) {
        foreach ($a in $NValues) {
            Write-Host "No. of trajectories = $a ($problemName, $solver)"
            Invoke-Point -ProblemName $problemName -Solver $solver -Nt $a
        }
    }
}

Pop-Location
