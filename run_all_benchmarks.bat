@echo off
REM Script to run all GPU ODE benchmarks in sequence
REM This allows for set-and-forget benchmarking while the GPU is available

REM Run from the repo root regardless of the caller's working directory
pushd "%~dp0"

echo =========================================
echo Starting All GPU ODE Benchmarks
echo =========================================
echo.

REM Parse command line arguments for custom nmax
set nmax_arg=
:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="-n" (
    set nmax_arg=-n %~2
    shift
    shift
    goto parse_args
)
echo Unknown option %~1
echo Usage: %~nx0 [-n nmax]
exit /b 1
:end_parse

REM Array of languages to benchmark
set languages=julia cpp pytorch jax cubie cubie_mlir

REM Run benchmarks for each language
for %%l in (%languages%) do (
    echo =========================================
    echo Benchmarking: %%l
    echo =========================================
    
    call "%~dp0run_benchmark.bat" -l %%l -d gpu -m ode %nmax_arg%
    if errorlevel 1 (
        echo.
        echo X Error occurred while benchmarking %%l
        echo Continuing with next language...
        echo.
    ) else (
        echo.
        echo Successfully completed benchmarking for %%l
        echo.
    )
)

echo =========================================
echo All Benchmarks Completed
echo =========================================
popd
