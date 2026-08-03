# PowerShell script for running C++ (MPGOS) ODE benchmarks
#
# The MPGOS Makefile is GNU-make syntax, which nmake cannot parse, and nvcc
# on Windows needs MSVC's cl.exe as its host compiler. Instead of requiring
# make, this script enters the Visual Studio developer environment (located
# via vswhere) and invokes nvcc directly with the Makefile's flags.
param(
    # Upper bound of the N sweep, or the literal "wp" for work-precision mode.
    [Parameter(Position=0)]
    [string]$MaxA,
    # classical-rk4 (RK4 build), cash-karp-54 (RKCK45 build), or all.
    [Parameter(Position=1)]
    [string]$Algorithm = 'all',
    # Build the requested solvers at NT=32768 and sweep dt/tolerance.
    [switch]$Wp
)

$ErrorActionPreference = 'Stop'

if ($MaxA -eq 'wp') { $Wp = $true }

[int]$MaxTrajectories = 0
if (-not $Wp -and -not [int]::TryParse($MaxA, [ref]$MaxTrajectories)) {
    Write-Error "Usage: run_ode_cpp.ps1 <max-trajectories>|wp [algorithm|all]"
}

# MPGOS solvers: RK4 (classical-rk4, fixed) and RKCK45 (cash-karp-54, adaptive).
$RunRk4 = $false
$RunRkck45 = $false
switch ($Algorithm) {
    'all' { $RunRk4 = $true; $RunRkck45 = $true }
    'classical-rk4' { $RunRk4 = $true }
    'cash-karp-54' { $RunRkck45 = $true }
    default {
        Write-Host "MPGOS does not support algorithm '$Algorithm'; skipping."
        exit 0
    }
}

# Load modules eagerly so the first-launch cubin load stays out of timed regions.
$env:CUDA_MODULE_LOADING = 'EAGER'

# Run from the repo root regardless of the caller's working directory
Push-Location (Join-Path $PSScriptRoot '..\..')

function Enter-VsEnvironment {
    if (Get-Command cl -ErrorAction SilentlyContinue) {
        return  # already in a developer shell
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

function Build-Project {
    # Mirrors GPU_ODE_MPGOS/Makefile. --gpu-architecture=native targets the
    # local GPU (sm_70 was dropped in CUDA 13). MSVC does not support
    # -std=c++11, so c++17 is used on Windows.
    if (Test-Path "GPU_ODE_MPGOS\Lorenz.exe") {
        Remove-Item "GPU_ODE_MPGOS\Lorenz.exe" -Force
    }
    nvcc -o GPU_ODE_MPGOS\Lorenz.exe GPU_ODE_MPGOS\Lorenz.cu `
        -I"GPU_ODE_MPGOS\SourceCodes" `
        -O3 -std=c++17 --ptxas-options=-v --gpu-architecture=native `
        -lineinfo -maxrregcount=128
    if ($LASTEXITCODE -ne 0) {
        Write-Error "nvcc build failed with exit code $LASTEXITCODE"
    }
}

# Lorenz.cu's config block is rewritten by absolute line number.
function Set-SolverConfig {
    param([string]$Solver, [int]$Nt)
    $content = Get-Content "GPU_ODE_MPGOS\Lorenz.cu"
    $content[14] = "#define SOLVER $Solver"
    $content[16] = "const int NT = $Nt;"
    $content | Set-Content "GPU_ODE_MPGOS\Lorenz.cu"
}

Enter-VsEnvironment

if ($Wp) {
    if ($RunRk4) {
        # RK4 build -> fixed-dt sweep
        Set-SolverConfig -Solver 'RK4' -Nt 32768
        Build-Project
        & "GPU_ODE_MPGOS\Lorenz.exe" 32768 wp
        if ($LASTEXITCODE -ne 0) { Write-Error "Lorenz.exe (RK4 wp) failed with exit code $LASTEXITCODE" }
    }

    if ($RunRkck45) {
        # RKCK45 build -> adaptive-tolerance sweep
        Set-SolverConfig -Solver 'RKCK45' -Nt 32768
        Build-Project
        & "GPU_ODE_MPGOS\Lorenz.exe" 32768 wp
        if ($LASTEXITCODE -ne 0) { Write-Error "Lorenz.exe (RKCK45 wp) failed with exit code $LASTEXITCODE" }
    }

    Pop-Location
    return
}

$a = 8

while ($a -le $MaxTrajectories) {
    Write-Host $a

    if ($RunRk4) {
        Set-SolverConfig -Solver 'RK4' -Nt $a
        Build-Project
        & "GPU_ODE_MPGOS\Lorenz.exe" $a
        if ($LASTEXITCODE -ne 0) { Write-Error "Lorenz.exe (RK4) failed with exit code $LASTEXITCODE" }
    }

    if ($RunRkck45) {
        Set-SolverConfig -Solver 'RKCK45' -Nt $a
        Build-Project
        & "GPU_ODE_MPGOS\Lorenz.exe" $a
        if ($LASTEXITCODE -ne 0) { Write-Error "Lorenz.exe (RKCK45) failed with exit code $LASTEXITCODE" }
    }

    # Increment the value
    $a = $a * 4
}

Pop-Location
