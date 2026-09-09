@echo off
setlocal enabledelayedexpansion
REM Forwards to `bench.py run` with the same flags and the clocks unlocked; -p accepts hyphenated package names.
pushd "%~dp0"
set "PASS="
:parse_loop
if "%~1"=="" goto parse_done
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
python bench.py run --no-lock-clocks !PASS!
set "STATUS=!errorlevel!"
popd
endlocal & exit /b %STATUS%
