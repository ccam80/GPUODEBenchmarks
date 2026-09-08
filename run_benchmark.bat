@echo off
setlocal enabledelayedexpansion
REM GPU ODE runs forward to bench.py with the same flags; -d cpu and -m sde reach the legacy runners under runner_scripts\<device>\.
pushd "%~dp0"
set "DEVICE=gpu"
set "MODEL=ode"
set "PACKAGE="
set "PASS="
:parse_loop
if "%~1"=="" goto parse_done
if /i "%~1"=="-d" ( set "DEVICE=%~2" & shift & shift & goto parse_loop )
if /i "%~1"=="--device" ( set "DEVICE=%~2" & shift & shift & goto parse_loop )
if /i "%~1"=="-m" ( set "MODEL=%~2" & shift & shift & goto parse_loop )
if /i "%~1"=="--model" ( set "MODEL=%~2" & shift & shift & goto parse_loop )
if /i "%~1"=="-p" goto set_package
if /i "%~1"=="--package" goto set_package
set "PASS=!PASS! %1"
shift
goto parse_loop
:set_package
set "PACKAGE=%~2"
set "PACKAGE=!PACKAGE:-=_!"
set "PASS=!PASS! -p !PACKAGE!"
shift
shift
goto parse_loop
:parse_done
if /i "%DEVICE%"=="gpu" if /i "%MODEL%"=="ode" (
    python bench.py --no-lock-clocks !PASS!
    set "STATUS=!errorlevel!"
    popd
    endlocal & exit /b %STATUS%
)
if "%PACKAGE%"=="" (
    echo -p/--package is required
    popd
    exit /b 1
)
set "RUNNER=runner_scripts\%DEVICE%\run_%MODEL%_%PACKAGE%.bat"
if not exist "%RUNNER%" (
    echo Ensemble %MODEL% on %DEVICE% with %PACKAGE% is not supported.
    popd
    exit /b 1
)
set "LEGACY=!PASS: -p %PACKAGE%=!"
call "%RUNNER%" !LEGACY!
set "STATUS=!errorlevel!"
popd
endlocal & exit /b %STATUS%
