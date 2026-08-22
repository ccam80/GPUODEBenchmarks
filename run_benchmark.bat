@echo off
setlocal enabledelayedexpansion

REM Generate benchmark data for one package and one analysis.
REM   -p, --package   julia | cpp | pytorch | jax | cubie | cubie_mlir | myokit_cuda
REM   -a, --analysis  performance (default) | work-precision | states | warm
REM   -n, --nmax      sweep ceiling (8, 32, ... <= n; default 16777216) or comma list of exact Ns
REM   -g, --algorithm all (default) | comma list of the names in runner_scripts/algorithms.csv
REM   -s, --problem   all (default) | comma list of names from runner_scripts\problems.csv
REM   -d, --device    gpu (default) | cpu
REM   -m, --model     ode (default) | sde
REM   --keep          keep existing output files (no pre-run deletion)
REM   --resume        skip every point already recorded on disk; implies --keep
REM   --resume-from   problem[:algorithm][:fixed|adaptive][:N] run-order cursor; skips everything before it; implies --keep

pushd "%~dp0"

set PACKAGE=
set ANALYSIS=performance
set NMAX=16777216
set ALGORITHM=all
set PROBLEM=all
set DEVICE=gpu
set MODEL=ode
set KEEP=
set RESUME=
set RESUME_FROM=

REM cmd splits unquoted commas into arguments; rejoin value tokens until the next -flag.
:parse_loop
if "%~1"=="" goto parse_done
if /i "%~1"=="--keep" (
    set "KEEP=1"
    shift
    goto parse_loop
)
if /i "%~1"=="--resume" (
    set "RESUME=1"
    set "KEEP=1"
    shift
    goto parse_loop
)
set "PA_TARGET="
if /i "%~1"=="--resume-from" set "PA_TARGET=RESUME_FROM"
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
if /i "%~1"=="-d" set "PA_TARGET=DEVICE"
if /i "%~1"=="--device" set "PA_TARGET=DEVICE"
if /i "%~1"=="-m" set "PA_TARGET=MODEL"
if /i "%~1"=="--model" set "PA_TARGET=MODEL"
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

REM Accept hyphenated aliases for underscore-separated package names
if /i "%PACKAGE%"=="cubie-mlir" set PACKAGE=cubie_mlir
if /i "%PACKAGE%"=="myokit-cuda" set PACKAGE=myokit_cuda

REM The continuation contract read by the bench scripts (runner_scripts/resume.py).
if defined RESUME_FROM set "KEEP=1"
if defined RESUME set "BENCH_RESUME=1"
if defined RESUME_FROM set "BENCH_RESUME_FROM=%RESUME_FROM%"

if "%PACKAGE%"=="" (
    echo -p/--package is required
    popd
    exit /b 1
)
if /i not "%DEVICE%"=="gpu" if /i not "%DEVICE%"=="cpu" (
    echo Unknown device "%DEVICE%" ^(gpu^|cpu^)
    popd
    exit /b 1
)
if /i not "%MODEL%"=="ode" if /i not "%MODEL%"=="sde" (
    echo Unknown model "%MODEL%" ^(ode^|sde^)
    popd
    exit /b 1
)
if /i not "%ANALYSIS%"=="performance" if /i not "%ANALYSIS%"=="work-precision" if /i not "%ANALYSIS%"=="states" if /i not "%ANALYSIS%"=="warm" (
    echo Unknown analysis "%ANALYSIS%" ^(performance^|work-precision^|states^|warm^)
    popd
    exit /b 1
)

REM -g: "all" or a comma list; the bench scripts reject unknown names.
set ALG_LIST=
set ALG_HAS_ALL=
for %%g in (!ALGORITHM!) do (
    set "TOK=%%g"
    if "!TOK!"=="all" ( set ALG_HAS_ALL=1 ) else ( set "ALG_LIST=!ALG_LIST! !TOK!" )
)
if defined ALG_HAS_ALL set "ALG_LIST=all"
if "!ALG_LIST!"=="" (
    echo -g/--algorithm requires a value
    popd
    exit /b 1
)

REM Problem names are validated by the frameworks against problems.csv.
if "!PROBLEM!"=="" (
    echo -s/--problem requires a value
    popd
    exit /b 1
)

if "!NMAX!"=="" (
    echo -n/--nmax must be a positive integer or a comma list of them
    popd
    exit /b 1
)
if "!NMAX:,=!"=="" (
    echo -n/--nmax must be a positive integer or a comma list of them, got "!NMAX!"
    popd
    exit /b 1
)
REM Sentinel keeps the variable defined when every digit and comma is stripped.
set "NMAX_CHECK=x!NMAX:,=!"
for %%d in (0 1 2 3 4 5 6 7 8 9) do set "NMAX_CHECK=!NMAX_CHECK:%%d=!"
if not "!NMAX_CHECK!"=="x" (
    echo -n/--nmax must be a positive integer or a comma list of them, got "!NMAX!"
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
    echo Unknown package: "%PACKAGE%". Supported: julia, cpp, jax, pytorch, cubie, cubie_mlir, myokit_cuda.
    popd
    exit /b 1
)

set "RUNNER=runner_scripts\%DEVICE%\run_%MODEL%_%PACKAGE%.bat"
if not exist "%RUNNER%" (
    echo Ensemble %MODEL% on %DEVICE% with %PACKAGE% is not supported.
    popd
    exit /b 1
)

if /i "%DEVICE%"=="gpu" if /i "%MODEL%"=="ode" (
    for /f "usebackq delims=" %%K in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0runner_scripts\bench_key.ps1"`) do set "DATASET_KEY=%%K"
    if not exist "data\%DATA_DIR%\!DATASET_KEY!\" mkdir "data\%DATA_DIR%\!DATASET_KEY!"
)

REM One runner invocation per requested algorithm; a failure does not stop the rest.
set benchmark_exit=0
for %%g in (!ALG_LIST!) do (
    echo Benchmarking %PACKAGE% %DEVICE% ensemble %MODEL% solvers ^(%ANALYSIS%, %%g, %PROBLEM%^)...

    REM Clear this machine's appended files for the analysis, algorithm and problems being run.
    if "%%g"=="all" (
        set "ALG_GLOB=*"
    ) else (
        set "ALG_GLOB=*_%%g"
    )
    if /i "%PROBLEM%"=="all" (
        set "PROBLEM_DIRS=*"
    ) else (
        set "PROBLEM_DIRS=!PROBLEM:,= !"
    )
    REM A bare * would expand to file names, so all-problems walks the key directory.
    if not defined KEEP if /i "%DEVICE%"=="gpu" if /i "%MODEL%"=="ode" (
        if /i "%PROBLEM%"=="all" (
            for /d %%d in ("data\%DATA_DIR%\!DATASET_KEY!\*") do call :clear_dir "%%d" "!ALG_GLOB!"
        ) else (
            for %%d in (!PROBLEM_DIRS!) do call :clear_dir "data\%DATA_DIR%\!DATASET_KEY!\%%d" "!ALG_GLOB!"
        )
    )

    call "%RUNNER%" -a %ANALYSIS% -n "%NMAX%" -g %%g -s "%PROBLEM%"
    if !errorlevel! neq 0 set benchmark_exit=1
)
popd
endlocal & exit /b %benchmark_exit%

REM Delete one problem directory's files for the analysis being run.
:clear_dir
if /i "%ANALYSIS%"=="work-precision" (
    del /q "%~1\*_wp_%~2.txt" 2>nul
) else if /i "%ANALYSIS%"=="states" (
    del /q "%~1\*_states_%~2.txt" 2>nul
) else if /i "%ANALYSIS%"=="warm" (
    rem warm deletes nothing
) else (
    del /q "%~1\*_times_%~2.txt" 2>nul
)
exit /b 0
