@echo off
setlocal enabledelayedexpansion

REM Generate benchmark data for one package and one analysis.
REM   -p, --package   julia | cpp | pytorch | jax | cubie | cubie_mlir | myokit_cuda
REM   -a, --analysis  performance (default) | work-precision
REM   -n, --nmax      largest trajectory count for a performance sweep (default 16777216)
REM   -d, --device    gpu (default) | cpu
REM   -m, --model     ode (default) | sde

pushd "%~dp0"

set PACKAGE=
set ANALYSIS=performance
set NMAX=16777216
set DEVICE=gpu
set MODEL=ode

:parse_loop
if "%~1"=="" goto parse_done
if /i "%~1"=="-p" (
    set PACKAGE=%~2
    shift
    shift
    goto parse_loop
)
if /i "%~1"=="--package" (
    set PACKAGE=%~2
    shift
    shift
    goto parse_loop
)
if /i "%~1"=="-a" (
    set ANALYSIS=%~2
    shift
    shift
    goto parse_loop
)
if /i "%~1"=="--analysis" (
    set ANALYSIS=%~2
    shift
    shift
    goto parse_loop
)
if /i "%~1"=="-n" (
    set NMAX=%~2
    shift
    shift
    goto parse_loop
)
if /i "%~1"=="--nmax" (
    set NMAX=%~2
    shift
    shift
    goto parse_loop
)
if /i "%~1"=="-d" (
    set DEVICE=%~2
    shift
    shift
    goto parse_loop
)
if /i "%~1"=="--device" (
    set DEVICE=%~2
    shift
    shift
    goto parse_loop
)
if /i "%~1"=="-m" (
    set MODEL=%~2
    shift
    shift
    goto parse_loop
)
if /i "%~1"=="--model" (
    set MODEL=%~2
    shift
    shift
    goto parse_loop
)
echo Unknown option %~1
popd
exit /b 1
:parse_done

REM Accept hyphenated aliases for underscore-separated package names
if /i "%PACKAGE%"=="cubie-mlir" set PACKAGE=cubie_mlir
if /i "%PACKAGE%"=="myokit-cuda" set PACKAGE=myokit_cuda

if "%PACKAGE%"=="" (
    echo -p/--package is required
    popd
    exit /b 1
)
if /i not "%ANALYSIS%"=="performance" if /i not "%ANALYSIS%"=="work-precision" (
    echo Unknown analysis "%ANALYSIS%" ^(performance^|work-precision^)
    popd
    exit /b 1
)

set DATA_DIR=
if /i "%PACKAGE%"=="julia" set DATA_DIR=Julia
if /i "%PACKAGE%"=="cpp" set DATA_DIR=CPP
if /i "%PACKAGE%"=="jax" set DATA_DIR=JAX
if /i "%PACKAGE%"=="pytorch" set DATA_DIR=PYTORCH
if /i "%PACKAGE%"=="cubie" set DATA_DIR=CUBIE
if /i "%PACKAGE%"=="cubie_mlir" set DATA_DIR=CUBIE_MLIR
if /i "%PACKAGE%"=="myokit_cuda" set DATA_DIR=MYOKIT_CUDA
if "%DATA_DIR%"=="" (
    echo Unknown package: %PACKAGE%. Supported: julia, cpp, jax, pytorch, cubie, cubie_mlir, myokit_cuda.
    popd
    exit /b 1
)

set RUNNER=runner_scripts\%DEVICE%\run_%MODEL%_%PACKAGE%.bat
if not exist "%RUNNER%" (
    echo Ensemble %MODEL% on %DEVICE% with %PACKAGE% is not supported.
    popd
    exit /b 1
)

echo Benchmarking %PACKAGE% %DEVICE% ensemble %MODEL% solvers ^(%ANALYSIS%^)...

REM Timing files are appended across the N sweep, so clear this machine's files
REM for the analysis being run; other machines' keyed files stay.
if /i "%DEVICE%"=="gpu" if /i "%MODEL%"=="ode" (
    for /f "usebackq delims=" %%K in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0runner_scripts\bench_key.ps1"`) do set "DATASET_KEY=%%K"
    if not exist "data\%DATA_DIR%\" mkdir "data\%DATA_DIR%"
    if /i "%ANALYSIS%"=="work-precision" (
        del /q "data\%DATA_DIR%\*_wp_*_!DATASET_KEY!.txt" 2>nul
    ) else (
        del /q "data\%DATA_DIR%\*_times_*_!DATASET_KEY!.txt" 2>nul
    )
)

call "%RUNNER%" -a %ANALYSIS% -n %NMAX%
set benchmark_exit=%errorlevel%
popd
endlocal & exit /b %benchmark_exit%
