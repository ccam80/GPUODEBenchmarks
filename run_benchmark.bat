@echo off
setlocal enabledelayedexpansion

REM Run from the repo root regardless of the caller's working directory
pushd "%~dp0"

REM Parse command line arguments
set lang=
set dev=
set model=
set nmax=
set has_n_option=false
set wp=false
set alg=all

:parse_loop
if "%~1"=="" goto end_parse_loop
if /i "%~1"=="-l" (
    set lang=%~2
    shift
    shift
    goto parse_loop
)
if /i "%~1"=="-d" (
    set dev=%~2
    shift
    shift
    goto parse_loop
)
if /i "%~1"=="-m" (
    set model=%~2
    shift
    shift
    goto parse_loop
)
if /i "%~1"=="-n" (
    set nmax=%~2
    set has_n_option=true
    shift
    shift
    goto parse_loop
)
if /i "%~1"=="-g" (
    set alg=%~2
    shift
    shift
    goto parse_loop
)
if /i "%~1"=="-w" (
    set wp=true
    shift
    goto parse_loop
)
echo Unknown option %~1
exit /b 1
:end_parse_loop

REM Set default nmax if not specified
if "%has_n_option%"=="false" (
    set /a nmax=16777216
)

REM Work-precision mode (-w): pass "wp" to the runner instead of nmax; the
REM runner sweeps its supported step size and/or tolerance controls at N=32768
REM against the golden reference.
if "%wp%"=="true" set nmax=wp

REM Accept hyphenated aliases for underscore-separated language names
if /i "%lang%"=="cubie-mlir" set lang=cubie_mlir
if /i "%lang%"=="myokit-cuda" set lang=myokit_cuda

REM Per-machine dataset key ("<os>_<gpu>"). Timing files are appended across the
REM N-sweep, so we clear only *this machine's* files before a run; other machines'
REM keyed files are left in place so data accumulates additively across machines.
set "DATASET_KEY="
for /f "usebackq delims=" %%K in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0runner_scripts\bench_key.ps1"`) do set "DATASET_KEY=%%K"

REM -g runs one algorithm (default "all"); the pre-run wipe narrows to match.
if /i "%alg%"=="all" (
    set "times_glob=*_times_*_%DATASET_KEY%.txt"
    set "wp_glob=*_wp_*_%DATASET_KEY%.txt"
) else (
    set "times_glob=*_times_*_%alg%_%DATASET_KEY%.txt"
    set "wp_glob=*_wp_*_%alg%_%DATASET_KEY%.txt"
)

echo %lang%

if /i "%lang%"=="julia" (
    echo Benchmarking Julia %dev% accelerated ensemble %model% solvers...
    if /i "%dev%"=="cpu" (
        call runner_scripts\%dev%\run_%model%_%lang%.bat %nmax%
    ) else if /i "%model%"=="sde" (
        call runner_scripts\%dev%\run_%model%_%lang%.bat %nmax%
    ) else (
        if not exist "data\Julia\" mkdir "data\Julia"
        if "%wp%"=="true" (
            del /q "data\Julia\%wp_glob%" 2>nul
        ) else (
            del /q "data\Julia\%times_glob%" 2>nul
        )
        call runner_scripts\%dev%\run_%model%_%lang%.bat %nmax% %alg%
    )
) else if /i "%lang%"=="jax" (
    goto check_ode_gpu
) else if /i "%lang%"=="pytorch" (
    goto check_ode_gpu
) else if /i "%lang%"=="cpp" (
    goto check_ode_gpu
) else if /i "%lang%"=="cubie" (
    goto check_ode_gpu
) else if /i "%lang%"=="cubie_mlir" (
    goto check_ode_gpu
) else if /i "%lang%"=="myokit_cuda" (
    goto check_ode_gpu
) else (
    echo Unknown language: %lang%. Supported: julia, cpp, jax, pytorch, cubie, cubie_mlir, myokit_cuda.
    popd
    exit /b 1
)

goto end_script

:check_ode_gpu
if /i not "%model%"=="ode" goto unsupported
if /i not "%dev%"=="gpu" goto unsupported

REM Convert language name to uppercase for data folder
set data_lang=%lang%
if /i "%lang%"=="jax" set data_lang=JAX
if /i "%lang%"=="pytorch" set data_lang=PYTORCH
if /i "%lang%"=="cpp" set data_lang=CPP
if /i "%lang%"=="cubie" set data_lang=CUBIE
if /i "%lang%"=="cubie_mlir" set data_lang=CUBIE_MLIR
if /i "%lang%"=="myokit_cuda" set data_lang=MYOKIT_CUDA

echo Benchmarking %lang% %dev% accelerated ensemble %model% solvers...
if not exist "data\%data_lang%\" mkdir "data\%data_lang%"
if "%wp%"=="true" (
    del /q "data\%data_lang%\%wp_glob%" 2>nul
) else (
    del /q "data\%data_lang%\%times_glob%" 2>nul
)
call runner_scripts\%dev%\run_%model%_%lang%.bat %nmax% %alg%
goto end_script

:unsupported
echo The benchmarking of ensemble %model% solvers on %dev% with %lang% is not supported.
echo Please use -m flag with "ode" and -d with "gpu".
popd
exit /b 1

:end_script
set benchmark_exit=%errorlevel%
popd
endlocal & exit /b %benchmark_exit%
