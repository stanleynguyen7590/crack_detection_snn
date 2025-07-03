@echo off
setlocal enabledelayedexpansion

:: =============================================================================
:: Crack Detection Training Runner (Live Output Version)
:: Batch script to run SNN and CNN baseline training with live console output
:: =============================================================================

echo.
echo ================================================================================
echo                    CRACK DETECTION TRAINING RUNNER (LIVE)
echo ================================================================================
echo.

:: Default parameters
set DATA_DIR=
set RUN_TYPE=both
set QUICK_MODE=
set TIME_STEPS=10

:: Parse command line arguments
:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-h" goto show_help
if /i "%~1"=="--data-dir" (
    set DATA_DIR=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--quick" (
    set QUICK_MODE=true
    shift
    goto parse_args
)
if /i "%~1"=="--time-steps" (
    set TIME_STEPS=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="-T" (
    set TIME_STEPS=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="snn" (
    set RUN_TYPE=snn
    shift
    goto parse_args
)
if /i "%~1"=="cnn" (
    set RUN_TYPE=cnn
    shift
    goto parse_args
)
if /i "%~1"=="both" (
    set RUN_TYPE=both
    shift
    goto parse_args
)
if /i "%~1"=="all" (
    set RUN_TYPE=both
    shift
    goto parse_args
)
shift
goto parse_args

:end_parse

:: Check if data directory is provided
if "%DATA_DIR%"=="" (
    echo ERROR: --data-dir is required!
    echo Use --help for usage information.
    echo.
    exit /b 1
)

:: Display configuration
echo Configuration:
echo   Data Directory: %DATA_DIR%
echo   Run Type:       %RUN_TYPE%
echo   Time Steps:     %TIME_STEPS% (SNN only)
if "%QUICK_MODE%"=="true" echo   Quick Mode:     Enabled (5 epochs, small batch)
echo.

:: Set parameters based on mode
if "%QUICK_MODE%"=="true" (
    set SNN_PARAMS=--batch-size 4 --num-epochs 5 --learning-rate 0.01 --time-steps %TIME_STEPS%
    set CNN_PARAMS=--batch-size 8 --num-epochs 5 --learning-rate 0.01 --skip-inception
    echo Quick mode: Using reduced epochs and batch sizes for fast testing
) else (
    set SNN_PARAMS=--batch-size 16 --num-epochs 10 --learning-rate 0.001 --time-steps %TIME_STEPS%
    set CNN_PARAMS=--batch-size 16 --num-epochs 10 --learning-rate 0.001
    echo Full training mode: Using recommended parameters
)
echo.

:: Run based on type selection
if /i "%RUN_TYPE%"=="snn" goto run_snn
if /i "%RUN_TYPE%"=="cnn" goto run_cnn
if /i "%RUN_TYPE%"=="both" goto run_both

:run_snn
echo.
echo ================================================================================
echo                           RUNNING SPIKING NEURAL NETWORK
echo ================================================================================
echo.

set SNN_CMD=python spiking_concrete.py --data-dir "%DATA_DIR%" %SNN_PARAMS%

echo Command: %SNN_CMD%
echo.
echo Starting SNN training at %time%...
echo.

:: Run SNN training with live output
%SNN_CMD%

if !errorlevel! equ 0 (
    echo.
    echo ✅ SNN training completed successfully!
) else (
    echo.
    echo ❌ SNN training failed with error code !errorlevel!
)

if /i "%RUN_TYPE%"=="snn" goto end_script
goto run_cnn

:run_cnn
echo.
echo ================================================================================
echo                           RUNNING CNN BASELINES
echo ================================================================================
echo.

set CNN_CMD=python baseline_concrete.py --data-dir "%DATA_DIR%" %CNN_PARAMS%

echo Command: %CNN_CMD%
echo.
echo Starting CNN baseline training at %time%...
echo.

:: Run CNN training with live output
%CNN_CMD%

if !errorlevel! equ 0 (
    echo.
    echo ✅ CNN baseline training completed successfully!
) else (
    echo.
    echo ❌ CNN baseline training failed with error code !errorlevel!
)

goto end_script

:run_both
echo Will run both SNN and CNN baseline training sequentially...
echo.

call :run_snn
echo.
echo ================================================================================
echo                      SNN COMPLETED - STARTING CNN BASELINES
echo ================================================================================
call :run_cnn

goto end_script

:show_help
echo.
echo USAGE:
echo   run_training_live.bat [TYPE] --data-dir PATH [OPTIONS]
echo.
echo REQUIRED:
echo   --data-dir PATH          Path to SDNET2018 dataset directory
echo.
echo TYPE (choose one):
echo   snn                      Run only Spiking Neural Network training
echo   cnn                      Run only CNN baseline training  
echo   both / all               Run both SNN and CNN training (default)
echo.
echo OPTIONS:
echo   --time-steps NUM         SNN simulation time steps (default: 10)
echo   -T NUM                   Alias for --time-steps
echo   --quick                  Quick mode: 5 epochs, smaller batches (for testing)
echo   --help, -h               Show this help message
echo.
echo EXAMPLES:
echo   # Train both models with full settings
echo   run_training_live.bat both --data-dir C:\data\SDNET2018
echo.
echo   # Train SNN with custom timesteps
echo   run_training_live.bat snn --data-dir C:\data\SDNET2018 --time-steps 20
echo.
echo   # Quick test run (5 epochs, 4 timesteps)
echo   run_training_live.bat both --data-dir C:\data\SDNET2018 --quick --time-steps 4
echo.
echo   # Train only CNN baselines
echo   run_training_live.bat cnn --data-dir C:\data\SDNET2018
echo.
goto end_script

:end_script
echo.
echo ================================================================================
echo                                TRAINING COMPLETE
echo ================================================================================
echo.
echo Check the following directories for results:
echo   - checkpoints\         (model checkpoints in timestamped folders)
echo   - results\             (evaluation results in timestamped folders)
echo.
echo Completed at %time%
echo.
pause
exit /b 0