@echo off
setlocal enabledelayedexpansion

REM Generate benchmark data for every package, or one, across one or more analyses.
REM   -p, --package   all (default) | julia | cpp | pytorch | jax | cubie | cubie_mlir | myokit_cuda
REM   -a, --analysis  performance (default) | work-precision | numerical | all
REM   -n, --nmax      largest trajectory count for a performance sweep (default 16777216)

pushd "%~dp0"

set PACKAGE=all
set ANALYSIS=performance
set NMAX=16777216

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
echo Unknown option %~1
popd
exit /b 1
:parse_done

if /i not "%ANALYSIS%"=="performance" if /i not "%ANALYSIS%"=="work-precision" if /i not "%ANALYSIS%"=="numerical" if /i not "%ANALYSIS%"=="all" (
    echo Unknown analysis "%ANALYSIS%" ^(performance^|work-precision^|numerical^|all^)
    popd
    exit /b 1
)

set PACKAGES=julia cpp pytorch jax cubie cubie_mlir myokit_cuda
if /i not "%PACKAGE%"=="all" set PACKAGES=%PACKAGE%

if /i "%ANALYSIS%"=="performance" call :run_sweep performance
if /i "%ANALYSIS%"=="all" call :run_sweep performance
if /i "%ANALYSIS%"=="performance" call :plot plot_ode_comp.jl
if /i "%ANALYSIS%"=="all" call :plot plot_ode_comp.jl

if /i "%ANALYSIS%"=="work-precision" call :run_sweep work-precision
if /i "%ANALYSIS%"=="all" call :run_sweep work-precision
if /i "%ANALYSIS%"=="work-precision" call :plot plot_ode_wp.jl
if /i "%ANALYSIS%"=="all" call :plot plot_ode_wp.jl

if /i "%ANALYSIS%"=="numerical" call "%~dp0run_numerical_equivalence.bat"
if /i "%ANALYSIS%"=="all" call "%~dp0run_numerical_equivalence.bat"

echo --- Pairwise numerical comparison ---
if exist "GPU_ODE_CUBIE\venv\Scripts\python.exe" (
    call "GPU_ODE_CUBIE\venv\Scripts\python.exe" compare_numerical_results.py
    if errorlevel 1 echo Pairwise comparison failed
) else (
    echo GPU_ODE_CUBIE venv not found; skipping pairwise comparison
)

popd
endlocal
exit /b 0

:run_sweep
for %%p in (%PACKAGES%) do (
    echo =========================================
    echo %~1: %%p
    echo =========================================
    call "%~dp0run_benchmark.bat" -p %%p -a %~1 -n %NMAX% -d gpu -m ode
    if errorlevel 1 (
        echo Error during %~1 for %%p; continuing with the next package
    ) else (
        echo Completed %~1 for %%p
    )
    echo.
)
exit /b 0

:plot
echo --- Plot: %~1 ---
julia --project=. runner_scripts\plot\%~1
if errorlevel 1 echo Plot %~1 failed
exit /b 0
