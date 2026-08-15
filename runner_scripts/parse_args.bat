@echo off
REM Sets ANALYSIS, NMAX, NLIST, ALGORITHM and PROBLEM in the caller; -n takes a sweep ceiling or comma list, -s a problem name, comma list, or all.
set ANALYSIS=performance
set NMAX=16777216
set ALGORITHM=all
set PROBLEM=all
set NLIST=

REM Tokens split on spaces and commas; value tokens after a flag are comma-joined.
set "PA_TARGET="
set "PA_HAVE="
set "PA_ERR="
if not defined PA_RAW goto parse_args_done
for %%x in (!PA_RAW!) do (
    set "TOK=%%~x"
    call :pa_token
)
if defined PA_ERR exit /b 1
if defined PA_TARGET if not defined PA_HAVE (
    echo !PA_TARGET! requires a value
    exit /b 1
)
:parse_args_done

if /i "%ANALYSIS%"=="performance" goto check_nmax
if /i "%ANALYSIS%"=="work-precision" goto check_nmax
echo Unknown analysis "%ANALYSIS%" ^(performance^|work-precision^)
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

:pa_token
if defined PA_ERR exit /b 0
set "PA_FLAG="
if /i "!TOK!"=="-a" set "PA_FLAG=ANALYSIS"
if /i "!TOK!"=="--analysis" set "PA_FLAG=ANALYSIS"
if /i "!TOK!"=="-n" set "PA_FLAG=NMAX"
if /i "!TOK!"=="--nmax" set "PA_FLAG=NMAX"
if /i "!TOK!"=="-g" set "PA_FLAG=ALGORITHM"
if /i "!TOK!"=="--algorithm" set "PA_FLAG=ALGORITHM"
if /i "!TOK!"=="-s" set "PA_FLAG=PROBLEM"
if /i "!TOK!"=="--problem" set "PA_FLAG=PROBLEM"
if defined PA_FLAG (
    if defined PA_TARGET if not defined PA_HAVE (
        echo !PA_TARGET! requires a value
        set "PA_ERR=1"
        exit /b 0
    )
    set "PA_TARGET=!PA_FLAG!"
    set "PA_HAVE="
    exit /b 0
)
if "!TOK:~0,1!"=="-" (
    echo Unknown option "!TOK!"
    set "PA_ERR=1"
    exit /b 0
)
if not defined PA_TARGET (
    echo Unknown option "!TOK!"
    set "PA_ERR=1"
    exit /b 0
)
if defined PA_HAVE (
    for %%v in (!PA_TARGET!) do set "%%v=!%%v!,!TOK!"
) else (
    for %%v in (!PA_TARGET!) do set "%%v=!TOK!"
    set "PA_HAVE=1"
)
exit /b 0
