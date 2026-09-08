@echo off
setlocal enabledelayedexpansion
REM Forwards to `bench.py -a numerical`; -p all|julia|cubie, --controller, --algorithm and -s map onto its flags.
pushd "%~dp0"
set "ARGS=-a numerical --no-lock-clocks"
set "PKG=cubie,julia"
:parse_loop
if "%~1"=="" goto parse_done
if /i "%~1"=="-p" goto set_package
if /i "%~1"=="--package" goto set_package
if /i "%~1"=="--controller" ( set "ARGS=!ARGS! --controller %~2" & shift & shift & goto parse_loop )
if /i "%~1"=="--algorithm" ( set "ARGS=!ARGS! -g %~2" & shift & shift & goto parse_loop )
if /i "%~1"=="-s" ( set "ARGS=!ARGS! -s %~2" & shift & shift & goto parse_loop )
if /i "%~1"=="--problem" ( set "ARGS=!ARGS! -s %~2" & shift & shift & goto parse_loop )
echo Unknown option %~1
popd
exit /b 1
:set_package
if /i "%~2"=="all" ( set "PKG=cubie,julia" ) else if /i "%~2"=="julia" ( set "PKG=julia" ) else if /i "%~2"=="cubie" ( set "PKG=cubie" ) else (
    echo Unknown package "%~2" ^(all^|julia^|cubie^)
    popd
    exit /b 1
)
shift
shift
goto parse_loop
:parse_done
python bench.py !ARGS! -p !PKG!
set "STATUS=!errorlevel!"
popd
endlocal & exit /b %STATUS%
