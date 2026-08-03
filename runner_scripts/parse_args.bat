@echo off
REM Sets ANALYSIS, NMAX, NLIST and ALGORITHM in the caller: call "%~dp0..\parse_args.bat" %*
REM -n: single value = sweep ceiling (8, 32, ... <= n); comma list = exact Ns. NLIST holds the counts, NMAX the largest.
set ANALYSIS=performance
set NMAX=16777216
set ALGORITHM=all
set NLIST=

REM cmd splits unquoted commas into arguments; rejoin value tokens until the next -flag.
:parse_args_loop
if "%~1"=="" goto parse_args_done
set "PA_TARGET="
if /i "%~1"=="-a" set "PA_TARGET=ANALYSIS"
if /i "%~1"=="--analysis" set "PA_TARGET=ANALYSIS"
if /i "%~1"=="-n" set "PA_TARGET=NMAX"
if /i "%~1"=="--nmax" set "PA_TARGET=NMAX"
if /i "%~1"=="-g" set "PA_TARGET=ALGORITHM"
if /i "%~1"=="--algorithm" set "PA_TARGET=ALGORITHM"
if not defined PA_TARGET (
    echo Unknown option %~1
    exit /b 1
)
if "%~2"=="" (
    echo %~1 requires a value
    exit /b 1
)
set "!PA_TARGET!=%~2"
shift
shift
:parse_args_collect
if "%~1"=="" goto parse_args_done
set "PA_NEXT=%~1"
if "!PA_NEXT:~0,1!"=="-" goto parse_args_loop
for %%v in (!PA_TARGET!) do set "%%v=!%%v!,%~1"
shift
goto parse_args_collect
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
    echo -n/--nmax must be a positive integer or a comma list of them
    exit /b 1
)
if "!NMAX:,=!"=="" (
    echo -n/--nmax must be a positive integer or a comma list of them, got "!NMAX!"
    exit /b 1
)
REM Sentinel keeps the variable defined when every digit and comma is stripped.
set "NMAX_CHECK=x!NMAX:,=!"
for %%d in (0 1 2 3 4 5 6 7 8 9) do set "NMAX_CHECK=!NMAX_CHECK:%%d=!"
if not "!NMAX_CHECK!"=="x" (
    echo -n/--nmax must be a positive integer or a comma list of them, got "!NMAX!"
    exit /b 1
)
if "!NMAX:,=!"=="!NMAX!" goto nmax_sweep

REM Comma list: run exactly these Ns; NMAX becomes the largest of them.
set "NLIST=!NMAX:,= !"
set NMAXV=0
for %%n in (!NLIST!) do if %%n gtr !NMAXV! set "NMAXV=%%n"
set "NMAX=!NMAXV!"
exit /b 0

:nmax_sweep
set /a NCUR=8
:nmax_sweep_loop
if !NCUR! gtr !NMAX! exit /b 0
set "NLIST=!NLIST! !NCUR!"
set /a NCUR=NCUR*4
if !NCUR! lss 8 exit /b 0
goto nmax_sweep_loop
