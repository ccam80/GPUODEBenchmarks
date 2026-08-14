@echo off
setlocal enabledelayedexpansion

REM Generate benchmark data for one or more packages across one or more analyses.
REM   -p, --package   all (default) | comma list of julia | cpp | pytorch | jax | cubie | cubie_mlir | myokit_cuda
REM   -a, --analysis  performance (default) | comma list of performance | work-precision | numerical | all
REM   -n, --nmax      sweep ceiling (8, 32, ... <= n; default 16777216) or comma list of exact Ns
REM   -g, --algorithm all (default) | comma list of euler|classical-rk4|tsit5|cash-karp-54
REM   -s, --problem   all (default) | comma list of names from runner_scripts\problems.csv
REM
REM e.g. run_all_benchmarks.bat -p cubie,julia -a performance,work-precision -g euler,tsit5 -n 8388608,134217728

pushd "%~dp0"

set PACKAGE=all
set ANALYSIS=performance
set NMAX=16777216
set ALGORITHM=all
set PROBLEM=all

REM cmd splits unquoted commas into arguments; rejoin value tokens until the next -flag.
:parse_loop
if "%~1"=="" goto parse_done
set "PA_TARGET="
if /i "%~1"=="-p" set "PA_TARGET=PACKAGE"
if /i "%~1"=="--package" set "PA_TARGET=PACKAGE"
if /i "%~1"=="-a" set "PA_TARGET=ANALYSIS"
if /i "%~1"=="--analysis" set "PA_TARGET=ANALYSIS"
if /i "%~1"=="-n" set "PA_TARGET=NMAX"
if /i "%~1"=="--nmax" set "PA_TARGET=NMAX"
if /i "%~1"=="-g" set "PA_TARGET=ALGORITHM"
if /i "%~1"=="--algorithm" set "PA_TARGET=ALGORITHM"
if /i "%~1"=="-s" set "PA_TARGET=PROBLEM"
if /i "%~1"=="--problem" set "PA_TARGET=PROBLEM"
if not defined PA_TARGET (
    echo Unknown option "%~1"
    popd
    exit /b 1
)
if "%~2"=="" (
    echo "%~1" requires a value
    popd
    exit /b 1
)
set "!PA_TARGET!=%~2"
shift
shift
:parse_collect
if "%~1"=="" goto parse_done
set "PA_NEXT=%~1"
if "!PA_NEXT:~0,1!"=="-" goto parse_loop
for %%v in (!PA_TARGET!) do set "%%v=!%%v!,%~1"
shift
goto parse_collect
:parse_done

REM Validate every value before it reaches a command line.
set DO_PERF=
set DO_WP=
set DO_NE=
set ARG_BAD=
for %%t in (!ANALYSIS!) do (
    set "TOK=%%t"
    set TOK_OK=
    if /i "!TOK!"=="performance" ( set DO_PERF=1& set TOK_OK=1 )
    if /i "!TOK!"=="work-precision" ( set DO_WP=1& set TOK_OK=1 )
    if /i "!TOK!"=="numerical" ( set DO_NE=1& set TOK_OK=1 )
    if /i "!TOK!"=="all" ( set DO_PERF=1& set DO_WP=1& set DO_NE=1& set TOK_OK=1 )
    if not defined TOK_OK (
        echo Unknown analysis "!TOK!" ^(performance^|work-precision^|numerical^|all^)
        set ARG_BAD=1
    )
)
if not defined DO_PERF if not defined DO_WP if not defined DO_NE (
    echo -a/--analysis requires a value
    set ARG_BAD=1
)

set PACKAGES=
set HAS_ALL_PACKAGES=
set HAS_JULIA=
set HAS_CUBIE=
for %%p in (!PACKAGE!) do (
    set "TOK=%%p"
    if /i "!TOK!"=="cubie-mlir" set "TOK=cubie_mlir"
    if /i "!TOK!"=="myokit-cuda" set "TOK=myokit_cuda"
    set TOK_OK=
    if /i "!TOK!"=="all" ( set HAS_ALL_PACKAGES=1& set TOK_OK=1 )
    if /i "!TOK!"=="julia" ( set HAS_JULIA=1& set TOK_OK=1 )
    if /i "!TOK!"=="cubie" ( set HAS_CUBIE=1& set TOK_OK=1 )
    if /i "!TOK!"=="cpp" set TOK_OK=1
    if /i "!TOK!"=="pytorch" set TOK_OK=1
    if /i "!TOK!"=="jax" set TOK_OK=1
    if /i "!TOK!"=="cubie_mlir" set TOK_OK=1
    if /i "!TOK!"=="myokit_cuda" set TOK_OK=1
    if not defined TOK_OK (
        echo Unknown package "!TOK!" ^(all^|julia^|cpp^|pytorch^|jax^|cubie^|cubie_mlir^|myokit_cuda^)
        set ARG_BAD=1
    )
    if defined TOK_OK if not "!TOK!"=="all" set "PACKAGES=!PACKAGES! !TOK!"
)
if defined HAS_ALL_PACKAGES set "PACKAGES=julia cpp pytorch jax cubie cubie_mlir myokit_cuda"
if "!PACKAGES!"=="" (
    echo -p/--package requires a value
    set ARG_BAD=1
)

for %%g in (!ALGORITHM!) do (
    set "TOK=%%g"
    set TOK_OK=
    if "!TOK!"=="all" set TOK_OK=1
    if "!TOK!"=="euler" set TOK_OK=1
    if "!TOK!"=="classical-rk4" set TOK_OK=1
    if "!TOK!"=="tsit5" set TOK_OK=1
    if "!TOK!"=="cash-karp-54" set TOK_OK=1
    if not defined TOK_OK (
        echo Unknown algorithm "!TOK!" ^(all^|euler^|classical-rk4^|tsit5^|cash-karp-54^)
        set ARG_BAD=1
    )
)

if "!NMAX!"=="" (
    echo -n/--nmax must be a positive integer or a comma list of them
    set ARG_BAD=1
) else if "!NMAX:,=!"=="" (
    echo -n/--nmax must be a positive integer or a comma list of them, got "!NMAX!"
    set ARG_BAD=1
) else (
    REM Sentinel keeps the variable defined when every digit and comma is stripped.
    set "NMAX_CHECK=x!NMAX:,=!"
    for %%d in (0 1 2 3 4 5 6 7 8 9) do set "NMAX_CHECK=!NMAX_CHECK:%%d=!"
    if not "!NMAX_CHECK!"=="x" (
        echo -n/--nmax must be a positive integer or a comma list of them, got "!NMAX!"
        set ARG_BAD=1
    )
)

if defined ARG_BAD (
    popd
    exit /b 1
)

if defined DO_PERF call :run_sweep performance
if defined DO_PERF call :plot plot_ode_comp.jl
if defined DO_WP call :run_sweep work-precision
if defined DO_WP call :plot plot_ode_wp.jl
if defined DO_NE call :run_numerical

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
for %%p in (!PACKAGES!) do (
    echo =========================================
    echo %~1: %%p
    echo =========================================
    call "%~dp0run_benchmark.bat" -p %%p -a %~1 -n "%NMAX%" -g "%ALGORITHM%" -s "%PROBLEM%" -d gpu -m ode
    if !errorlevel! neq 0 (
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

:run_numerical
REM The numerical-equivalence suite only covers julia and cubie.
set NE_PACKAGE=
if defined HAS_ALL_PACKAGES set NE_PACKAGE=all
if not defined NE_PACKAGE if defined HAS_JULIA if defined HAS_CUBIE set NE_PACKAGE=all
if not defined NE_PACKAGE if defined HAS_JULIA set NE_PACKAGE=julia
if not defined NE_PACKAGE if defined HAS_CUBIE set NE_PACKAGE=cubie
if not defined NE_PACKAGE (
    echo Numerical equivalence skipped: no requested package is in the suite ^(all^|julia^|cubie^)
    exit /b 0
)
call "%~dp0run_numerical_equivalence.bat" -p %NE_PACKAGE% -s "%PROBLEM%"
exit /b 0
