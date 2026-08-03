@echo off
REM Sets ANALYSIS, NMAX and ALGORITHM in the caller: call "%~dp0..\parse_args.bat" %*
set ANALYSIS=performance
set NMAX=16777216
set ALGORITHM=all

:parse_args_loop
if "%~1"=="" goto parse_args_done
if /i "%~1"=="-a" (
    if "%~2"=="" (
        echo %~1 requires a value
        exit /b 1
    )
    set "ANALYSIS=%~2"
    shift
    shift
    goto parse_args_loop
)
if /i "%~1"=="--analysis" (
    if "%~2"=="" (
        echo %~1 requires a value
        exit /b 1
    )
    set "ANALYSIS=%~2"
    shift
    shift
    goto parse_args_loop
)
if /i "%~1"=="-n" (
    if "%~2"=="" (
        echo %~1 requires a value
        exit /b 1
    )
    set "NMAX=%~2"
    shift
    shift
    goto parse_args_loop
)
if /i "%~1"=="--nmax" (
    if "%~2"=="" (
        echo %~1 requires a value
        exit /b 1
    )
    set "NMAX=%~2"
    shift
    shift
    goto parse_args_loop
)
if /i "%~1"=="-g" (
    if "%~2"=="" (
        echo %~1 requires a value
        exit /b 1
    )
    set "ALGORITHM=%~2"
    shift
    shift
    goto parse_args_loop
)
if /i "%~1"=="--algorithm" (
    if "%~2"=="" (
        echo %~1 requires a value
        exit /b 1
    )
    set "ALGORITHM=%~2"
    shift
    shift
    goto parse_args_loop
)
echo Unknown option %~1
exit /b 1
:parse_args_done

if /i "%ANALYSIS%"=="performance" goto check_algorithm
if /i "%ANALYSIS%"=="work-precision" goto check_algorithm
echo Unknown analysis "%ANALYSIS%" ^(performance^|work-precision^)
exit /b 1

:check_algorithm
if "%ALGORITHM%"=="all" goto check_nmax
if "%ALGORITHM%"=="euler" goto check_nmax
if "%ALGORITHM%"=="classical-rk4" goto check_nmax
if "%ALGORITHM%"=="tsit5" goto check_nmax
if "%ALGORITHM%"=="cash-karp-54" goto check_nmax
echo Unknown algorithm "%ALGORITHM%" ^(all^|euler^|classical-rk4^|tsit5^|cash-karp-54^)
exit /b 1

:check_nmax
if "!NMAX!"=="" (
    echo -n/--nmax must be a positive integer
    exit /b 1
)
REM Sentinel keeps the variable defined when every digit is stripped.
set "NMAX_CHECK=x!NMAX!"
for %%d in (0 1 2 3 4 5 6 7 8 9) do set "NMAX_CHECK=!NMAX_CHECK:%%d=!"
if not "!NMAX_CHECK!"=="x" (
    echo -n/--nmax must be a positive integer, got "!NMAX!"
    exit /b 1
)
exit /b 0
