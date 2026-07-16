@echo off
REM Run the full numerical-equivalence (ne) suite: golden reference (if
REM missing), DifferentialEquations.jl Float32 reference sweeps, cubie
REM Float32 sweeps, and the comparison report + plots.
REM
REM Usage: run_numerical_equivalence.bat [fixed^|adaptive^|all]
REM   fixed    - error-vs-dt convergence sweeps only
REM   adaptive - error-vs-tolerance sweeps only (default + matched controllers)
REM   all      - both (default)
REM
REM Exit code: non-zero if any step fails or the comparison finds a
REM MISMATCH / DIVERGENT algorithm (compare_numerical_equivalence.py exits
REM 2) - suitable as a CI gate. The Julia outputs are machine independent;
REM if data\numerical_equivalence\julia\ is committed and Julia is
REM unavailable, run the last two steps by hand instead.

REM Run from the repo root regardless of the caller's working directory
pushd "%~dp0"

set mode=%~1
if "%mode%"=="" set mode=all
if /i not "%mode%"=="fixed" if /i not "%mode%"=="adaptive" if /i not "%mode%"=="all" (
    echo Usage: %~nx0 [fixed^|adaptive^|all]
    popd
    exit /b 1
)

echo =========================================
echo Numerical-equivalence suite (mode: %mode%)
echo =========================================

if not exist "data\numerical\golden_ne_lorenz_1024.csv" (
    echo --- Generating golden reference (Float64 Vern9, machine independent^) ---
    julia -t auto --project=. runner_scripts\numerical_equivalence\generate_golden_ne.jl
    if errorlevel 1 (
        echo X golden generation failed
        popd
        exit /b 1
    )
)

echo --- DifferentialEquations.jl Float32 sweeps (CPU, machine independent^) ---
julia -t auto --project=. runner_scripts\numerical_equivalence\ne_diffeq.jl %mode%
if errorlevel 1 (
    echo X DifferentialEquations.jl sweeps failed
    popd
    exit /b 1
)

echo --- cubie Float32 sweeps (GPU, keyed per machine^) ---
if not exist "GPU_ODE_CUBIE\venv\Scripts\python.exe" (
    echo X GPU_ODE_CUBIE venv not found; run setup_all_environments.py first
    popd
    exit /b 1
)
call "GPU_ODE_CUBIE\venv\Scripts\python.exe" GPU_ODE_CUBIE\numerical_equivalence.py %mode%
if errorlevel 1 (
    echo X cubie sweeps failed
    popd
    exit /b 1
)

echo --- Comparison report + plots ---
call "GPU_ODE_CUBIE\venv\Scripts\python.exe" compare_numerical_equivalence.py
set compare_status=%errorlevel%
if %compare_status%==0 (
    echo All algorithms equivalent/tracking
) else (
    echo X Comparison found mismatching algorithms (see numerical_equivalence_^<os^>_^<gpu^>.md^)
)

popd
exit /b %compare_status%
