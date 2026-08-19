@echo off
setlocal enabledelayedexpansion
set "PA_RAW=%*"
call "%~dp0..\parse_args.bat"
if errorlevel 1 exit /b 1

REM One julia process per (problem, algorithm) runs that pair's whole N sweep.
set "ALGO_LIST="
for /f "usebackq delims=" %%g in (`python runner_scripts\algorithms.py julia "%ALGORITHM%"`) do (
    set "ALGO_LIST=!ALGO_LIST! %%g"
)
if "!ALGO_LIST!"=="" (
    echo Julia ^(DiffEqGPU kernel path^) runs none of the requested algorithms; skipping.
    endlocal
    exit /b 0
)

if /i "%ANALYSIS%"=="states" (
    REM Parallel compiles, serialized GPU sections; -n overrides the ensemble size.
    if not "%NMAX%"=="16777216" (
        python runner_scripts\gpu\julia_states_driver.py "%ALGORITHM%" "%NMAX%"
    ) else (
        python runner_scripts\gpu\julia_states_driver.py "%ALGORITHM%"
    )
    if errorlevel 1 exit /b 1
    endlocal
    exit /b 0
)

if /i "%ANALYSIS%"=="work-precision" (
    for %%g in (!ALGO_LIST!) do (
        julia --project=. GPU_ODE_Julia\bench_ode_gpu.jl wp "%%g" --problem "%PROBLEM%"
        if !errorlevel! neq 0 exit /b 1
    )
    endlocal
    exit /b 0
)

set "PROBLEM_LIST="
for /f "usebackq delims=" %%p in (`python runner_scripts\problems.py julia "%PROBLEM%"`) do (
    set "PROBLEM_LIST=!PROBLEM_LIST! %%p"
)
if "!PROBLEM_LIST!"=="" (
    echo Julia runs none of the requested problems; skipping.
    endlocal
    exit /b 0
)

set "NLIST_CSV=!NLIST: =,!"
if "!NLIST_CSV:~0,1!"=="," set "NLIST_CSV=!NLIST_CSV:~1!"

for %%p in (!PROBLEM_LIST!) do (
    for %%g in (!ALGO_LIST!) do (
        echo Problem %%p, algorithm %%g, N sweep = !NLIST_CSV!
        julia --project=. GPU_ODE_Julia\bench_ode_gpu.jl "!NLIST_CSV!" "%%g" --problem "%%p"
        if !errorlevel! neq 0 exit /b 1
    )
)

endlocal
