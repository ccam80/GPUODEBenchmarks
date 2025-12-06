@echo off
setlocal enabledelayedexpansion

REM Parse command line arguments
set lang=
set dev=
set model=
set nmax=
set has_n_option=false

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
echo Unknown option %~1
exit /b 1
:end_parse_loop

REM Set default nmax if not specified
if "%has_n_option%"=="false" (
    set /a nmax=16777216
)

echo %lang%

if /i "%lang%"=="julia" (
    echo Benchmarking Julia %dev% accelerated ensemble %model% solvers...
    if /i "%dev%"=="cpu" (
        call runner_scripts\%dev%\run_%model%_%lang%.bat %nmax%
    ) else if /i "%model%"=="sde" (
        call runner_scripts\%dev%\run_%model%_%lang%.bat %nmax%
    ) else (
        if exist "data\Julia\" (
            del /q "data\Julia\*" 2>nul
            if not exist "data\Julia\" mkdir "data\Julia"
        ) else (
            mkdir "data\Julia"
        )
        call runner_scripts\%dev%\run_%model%_%lang%.bat %nmax%
    )
) else if /i "%lang%"=="jax" goto check_ode_gpu
) else if /i "%lang%"=="pytorch" goto check_ode_gpu
) else if /i "%lang%"=="cpp" goto check_ode_gpu
) else if /i "%lang%"=="cubie" goto check_ode_gpu
) else (
    goto end_script
)

goto end_script

:check_ode_gpu
if not "%model%"=="ode" goto unsupported
if not "%dev%"=="gpu" goto unsupported

echo Benchmarking %lang% %dev% accelerated ensemble %model% solvers...
if exist "data\%lang%\" (
    del /q "data\%lang%\*" 2>nul
    if not exist "data\%lang%\" mkdir "data\%lang%"
) else (
    mkdir "data\%lang%"
)
call runner_scripts\%dev%\run_%model%_%lang%.bat %nmax%
goto end_script

:unsupported
echo The benchmarking of ensemble %model% solvers on %dev% with %lang% is not supported.
echo Please use -m flag with "ode" and -d with "gpu".
exit /b 1

:end_script
endlocal
