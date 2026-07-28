@echo off
REM Script to run all GPU ODE benchmarks in sequence
REM This allows for set-and-forget benchmarking while the GPU is available

REM Run from the repo root regardless of the caller's working directory
pushd "%~dp0"

echo =========================================
echo Starting All GPU ODE Benchmarks
echo =========================================
echo.

REM Parse command line arguments for custom nmax, work-precision and
REM numerical-precision modes.
REM -w also runs the work-precision (error-vs-time) sweeps and their plot.
REM -np/--numerical-precision also runs the numerical-equivalence suite: the
REM fixed-step error-vs-dt sweeps of every algorithm mutually supported by
REM cubie and DifferentialEquations.jl (both in Float32) plus their
REM comparison report.
set nmax_arg=
set wp=false
set np=false
:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="-n" (
    set nmax_arg=-n %~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="-w" (
    set wp=true
    shift
    goto parse_args
)
if /i "%~1"=="-np" (
    set np=true
    shift
    goto parse_args
)
if /i "%~1"=="--numerical-precision" (
    set np=true
    shift
    goto parse_args
)
echo Unknown option %~1
echo Usage: %~nx0 [-n nmax] [-w] [-np^|--numerical-precision]
exit /b 1
:end_parse

REM Array of languages to benchmark
set languages=julia cpp pytorch jax cubie cubie_mlir myokit_cuda

REM Run timing benchmarks for each language
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

REM Optionally run the work-precision sweeps for each language (-w).
if "%wp%"=="true" (
    for %%l in (%languages%) do (
        echo =========================================
        echo Work-precision benchmarking: %%l
        echo =========================================

        call "%~dp0run_benchmark.bat" -l %%l -d gpu -m ode -w
        if errorlevel 1 (
            echo.
            echo X Error occurred while work-precision benchmarking %%l
            echo Continuing with next language...
            echo.
        ) else (
            echo.
            echo Successfully completed work-precision benchmarking for %%l
            echo.
        )
    )
)

REM Optionally run the numerical-equivalence suite (-np/--numerical-precision):
REM Float32 fixed-step error-vs-dt sweeps of every mutually supported
REM algorithm, for DifferentialEquations.jl (CPU reference) and cubie (GPU),
REM then the comparison report + plot.
if "%np%"=="true" (
    call "%~dp0run_numerical_equivalence.bat"
    if errorlevel 1 (
        echo X Numerical-equivalence suite reported problems (see numerical_equivalence_^<os^>_^<gpu^>.md^)
    ) else (
        echo Numerical-equivalence suite completed (all algorithms equivalent/tracking^)
    )
    echo.
)

echo =========================================
echo All Benchmarks Completed
echo =========================================
echo.

echo =========================================
echo Generating timing comparison plot
echo =========================================
julia --project=. runner_scripts\plot\plot_ode_comp.jl
if errorlevel 1 (
    echo X Error occurred while generating the timing comparison plot
) else (
    echo Plot saved to .\plots
)
echo.

REM Work-precision plot (only meaningful when -w regenerated the wp data).
if "%wp%"=="true" (
    echo =========================================
    echo Generating work-precision plot
    echo =========================================
    julia --project=. runner_scripts\plot\plot_ode_wp.jl
    if errorlevel 1 (
        echo X Error occurred while generating the work-precision plot
    ) else (
        echo Plot saved to .\plots
    )
    echo.
)

echo =========================================
echo Comparing numerical results
echo =========================================
if exist "GPU_ODE_CUBIE\venv\Scripts\python.exe" (
    call "GPU_ODE_CUBIE\venv\Scripts\python.exe" compare_numerical_results.py
    if errorlevel 1 (
        echo X Error occurred while comparing numerical results
    ) else (
        echo Numerical comparison written to .\pairwise_comparisons.md
    )
) else (
    echo X Could not find GPU_ODE_CUBIE\venv; skipping numerical comparison
)
echo.

popd
