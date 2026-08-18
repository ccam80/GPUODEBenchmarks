@echo off
setlocal enabledelayedexpansion
set "PA_RAW=%*"
call "%~dp0..\parse_args.bat"
if errorlevel 1 exit /b 1

REM One julia process per algorithm, so a watchdog exit only abandons that leg.
set "ALGO_LIST="
for /f "usebackq delims=" %%g in (`python runner_scripts\algorithms.py julia "%ALGORITHM%"`) do (
    set "ALGO_LIST=!ALGO_LIST! %%g"
)
if "!ALGO_LIST!"=="" (
    echo Julia ^(DiffEqGPU kernel path^) runs none of the requested algorithms; skipping.
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

for %%a in (!NLIST!) do (
    echo No. of trajectories = %%a
    for %%g in (!ALGO_LIST!) do (
        julia --project=. GPU_ODE_Julia\bench_ode_gpu.jl %%a "%%g" --problem "%PROBLEM%"
        if !errorlevel! neq 0 exit /b 1
    )
)

endlocal
