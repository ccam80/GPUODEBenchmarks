@echo off
setlocal enabledelayedexpansion

REM Run the numerical-equivalence analysis: golden reference (if missing),
REM DifferentialEquations.jl Float32 reference sweeps, cubie Float32 sweeps,
REM and the comparison report + plots.
REM   -p, --package     all (default) | julia | cubie
REM   --controller      all (default) | fixed | adaptive
REM   --algorithm       all (default) | a cubie alias from algorithms.csv

pushd "%~dp0"

set PACKAGE=all
set CONTROLLER=all
set ALGORITHM=all

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
if /i "%~1"=="--controller" (
    set CONTROLLER=%~2
    shift
    shift
    goto parse_loop
)
if /i "%~1"=="--algorithm" (
    set ALGORITHM=%~2
    shift
    shift
    goto parse_loop
)
echo Unknown option %~1
popd
exit /b 1
:parse_done

if /i not "%PACKAGE%"=="all" if /i not "%PACKAGE%"=="julia" if /i not "%PACKAGE%"=="cubie" (
    echo Unknown package "%PACKAGE%" ^(all^|julia^|cubie^)
    popd
    exit /b 1
)
if /i not "%CONTROLLER%"=="all" if /i not "%CONTROLLER%"=="fixed" if /i not "%CONTROLLER%"=="adaptive" (
    echo Unknown controller "%CONTROLLER%" ^(all^|fixed^|adaptive^)
    popd
    exit /b 1
)

echo =========================================
echo Numerical equivalence (package: %PACKAGE%, controller: %CONTROLLER%, algorithm: %ALGORITHM%)
echo =========================================

if not exist "GPU_ODE_CUBIE\venv\Scripts\python.exe" (
    echo GPU_ODE_CUBIE venv not found; run setup_all_environments.py first
    popd
    exit /b 1
)

if /i "%PACKAGE%"=="cubie" goto cubie_sweeps

if not exist "data\numerical\golden_ne_lorenz_1024.csv" (
    echo --- Golden reference ^(Float64 Vern9, machine independent^) ---
    julia -t auto --project=. runner_scripts\numerical_equivalence\generate_golden_ne.jl
    if errorlevel 1 (
        echo golden generation failed
        popd
        exit /b 1
    )
)

echo --- DifferentialEquations.jl Float32 sweeps ^(CPU, machine independent^) ---
julia -t auto --project=. runner_scripts\numerical_equivalence\ne_diffeq.jl --controller %CONTROLLER% --algorithm %ALGORITHM%
if errorlevel 1 (
    echo DifferentialEquations.jl sweeps failed
    popd
    exit /b 1
)

:cubie_sweeps
if /i "%PACKAGE%"=="julia" goto compare

echo --- cubie Float32 sweeps ^(GPU, keyed per machine^) ---
call "GPU_ODE_CUBIE\venv\Scripts\python.exe" GPU_ODE_CUBIE\numerical_equivalence.py --controller %CONTROLLER% --algorithm %ALGORITHM%
if errorlevel 1 (
    echo cubie sweeps failed
    popd
    exit /b 1
)

:compare
echo --- Comparison report + plots ---
call "GPU_ODE_CUBIE\venv\Scripts\python.exe" compare_numerical_equivalence.py
set compare_status=%errorlevel%
if %compare_status%==0 (
    echo All algorithms equivalent/tracking
) else (
    echo Comparison found mismatching algorithms ^(see numerical_equivalence_^<os^>_^<gpu^>.md^)
)

popd
exit /b %compare_status%
