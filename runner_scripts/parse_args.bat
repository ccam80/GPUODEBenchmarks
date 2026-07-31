@echo off
REM Sets ANALYSIS and NMAX in the caller: call "%~dp0..\parse_args.bat" %*
set ANALYSIS=performance
set NMAX=16777216

:parse_args_loop
if "%~1"=="" goto parse_args_done
if /i "%~1"=="-a" (
    if "%~2"=="" (
        echo %~1 requires a value
        exit /b 1
    )
    set ANALYSIS=%~2
    shift
    shift
    goto parse_args_loop
)
if /i "%~1"=="--analysis" (
    if "%~2"=="" (
        echo %~1 requires a value
        exit /b 1
    )
    set ANALYSIS=%~2
    shift
    shift
    goto parse_args_loop
)
if /i "%~1"=="-n" (
    if "%~2"=="" (
        echo %~1 requires a value
        exit /b 1
    )
    set NMAX=%~2
    shift
    shift
    goto parse_args_loop
)
if /i "%~1"=="--nmax" (
    if "%~2"=="" (
        echo %~1 requires a value
        exit /b 1
    )
    set NMAX=%~2
    shift
    shift
    goto parse_args_loop
)
echo Unknown option %~1
exit /b 1
:parse_args_done

if /i "%ANALYSIS%"=="performance" exit /b 0
if /i "%ANALYSIS%"=="work-precision" exit /b 0
echo Unknown analysis "%ANALYSIS%" ^(performance^|work-precision^)
exit /b 1
