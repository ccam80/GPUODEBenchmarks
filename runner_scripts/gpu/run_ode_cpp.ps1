# Windows MPGOS runner: enters the Visual Studio developer environment and calls nvcc.
param(
    [ValidateSet('performance', 'work-precision')]
    [string]$Analysis = 'performance',
    [long]$Nmax = 16777216
)

$ErrorActionPreference = 'Stop'

# Load modules eagerly so the first-launch cubin load stays out of timed regions.
$env:CUDA_MODULE_LOADING = 'EAGER'

Push-Location (Join-Path $PSScriptRoot '..\..')

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

# Solver and trajectory count are compile-time constants, so each point is a rebuild.
function Invoke-Point {
    param([string]$Solver, [long]$Nt, [switch]$Wp)
    $content = Get-Content "GPU_ODE_MPGOS\Lorenz.cu"
    $content[14] = "#define SOLVER $Solver"
    $content[16] = "const int NT = $Nt;"
    $content | Set-Content "GPU_ODE_MPGOS\Lorenz.cu"
    Build-Project
    if ($Wp) {
        & "GPU_ODE_MPGOS\Lorenz.exe" $Nt wp
    } else {
        & "GPU_ODE_MPGOS\Lorenz.exe" $Nt
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Lorenz.exe ($Solver) failed with exit code $LASTEXITCODE"
    }
}

Enter-VsEnvironment

if ($Analysis -eq 'work-precision') {
    foreach ($solver in @('RK4', 'RKCK45')) {
        Invoke-Point -Solver $solver -Nt 32768 -Wp
    }
    Pop-Location
    return
}

$a = 8
while ($a -le $Nmax) {
    Write-Host "No. of trajectories = $a"
    foreach ($solver in @('RK4', 'RKCK45')) {
        Invoke-Point -Solver $solver -Nt $a
    }
    $a = $a * 4
}

Pop-Location
