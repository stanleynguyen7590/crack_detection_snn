@echo off
setlocal enabledelayedexpansion

:: =============================================================================
:: Crack Detection Training Runner
:: Batch script to run SNN and CNN baseline training with convenient options
:: =============================================================================

echo.
echo ================================================================================
echo                    CRACK DETECTION TRAINING RUNNER
echo ================================================================================
echo.

:: Default parameters
set DATA_DIR=
set BATCH_SIZE=16
set EPOCHS=20
set LEARNING_RATE=0.001
set TIME_STEPS=10
set MODE=train
set DEVICE=auto
set NO_TIMESTAMPS=
set SKIP_INCEPTION=
set EXTRA_ARGS=

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
if /i "%~1"=="--batch-size" (
    set BATCH_SIZE=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--epochs" (
    set EPOCHS=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--learning-rate" (
    set LEARNING_RATE=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--lr" (
    set LEARNING_RATE=%~2
    shift
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
if /i "%~1"=="--mode" (
    set MODE=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--device" (
    set DEVICE=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--no-timestamps" (
    set NO_TIMESTAMPS=--no-timestamps
    shift
    goto parse_args
)
if /i "%~1"=="--skip-inception" (
    set SKIP_INCEPTION=--skip-inception
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
:: Capture any remaining arguments
set EXTRA_ARGS=!EXTRA_ARGS! %~1
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

:: Set default run type if not specified
if "%RUN_TYPE%"=="" set RUN_TYPE=both

:: Display configuration
echo Configuration:
echo   Data Directory: %DATA_DIR%
echo   Run Type:       %RUN_TYPE%
echo   Batch Size:     %BATCH_SIZE%
echo   Epochs:         %EPOCHS%
echo   Learning Rate:  %LEARNING_RATE%
echo   Time Steps:     %TIME_STEPS% (SNN only)
echo   Mode:           %MODE%
echo   Device:         %DEVICE%
if not "%NO_TIMESTAMPS%"=="" echo   No Timestamps:  Yes
if not "%SKIP_INCEPTION%"=="" echo   Skip Inception: Yes
if not "%EXTRA_ARGS%"=="" echo   Extra Args:     %EXTRA_ARGS%
echo.

:: Create timestamp for logging
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do if not "%%I"=="" set datetime=%%I
set timestamp=!datetime:~0,8!_!datetime:~8,6!

:: Create logs directory
if not exist logs mkdir logs

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

set SNN_CMD=python spiking_concrete.py --data-dir "%DATA_DIR%" --batch-size %BATCH_SIZE% --num-epochs %EPOCHS% --learning-rate %LEARNING_RATE% --time-steps %TIME_STEPS% --mode %MODE% --device %DEVICE% %NO_TIMESTAMPS% %EXTRA_ARGS%

echo Command: %SNN_CMD%
echo.
echo Starting SNN training at %time%...
echo.

:: Run SNN training with logging
%SNN_CMD% > logs\snn_training_!timestamp!.log 2>&1

if !errorlevel! equ 0 (
    echo.
    echo ✅ SNN training completed successfully!
    echo Log saved to: logs\snn_training_!timestamp!.log
    echo.
    echo Last few lines of output:
    powershell "Get-Content logs\snn_training_!timestamp!.log | Select-Object -Last 10"
) else (
    echo.
    echo ❌ SNN training failed with error code !errorlevel!
    echo Check log: logs\snn_training_!timestamp!.log
    echo.
    echo Error details:
    powershell "Get-Content logs\snn_training_!timestamp!.log | Select-Object -Last 20"
)

if /i "%RUN_TYPE%"=="snn" goto end_script
goto run_cnn

:run_cnn
echo.
echo ================================================================================
echo                           RUNNING CNN BASELINES
echo ================================================================================
echo.

:: Adjust batch size for CNN (usually can handle larger batches)
set CNN_BATCH_SIZE=%BATCH_SIZE%
if %BATCH_SIZE% lss 16 set CNN_BATCH_SIZE=16

:: Adjust epochs for CNN (usually needs fewer epochs with pretrained weights)
set CNN_EPOCHS=%EPOCHS%
if %EPOCHS% gtr 20 set CNN_EPOCHS=15

set CNN_CMD=python baseline_concrete.py --data-dir "%DATA_DIR%" --batch-size %CNN_BATCH_SIZE% --num-epochs %CNN_EPOCHS% --learning-rate %LEARNING_RATE% --device %DEVICE% %NO_TIMESTAMPS% %SKIP_INCEPTION% %EXTRA_ARGS%

echo Command: %CNN_CMD%
echo.
echo Starting CNN baseline training at %time%...
echo   Batch Size: %CNN_BATCH_SIZE% (adjusted for CNN)
echo   Epochs:     %CNN_EPOCHS% (adjusted for CNN)
echo.

:: Run CNN training with logging
%CNN_CMD% > logs\cnn_training_!timestamp!.log 2>&1

if !errorlevel! equ 0 (
    echo.
    echo ✅ CNN baseline training completed successfully!
    echo Log saved to: logs\cnn_training_!timestamp!.log
    echo.
    echo Last few lines of output:
    powershell "Get-Content logs\cnn_training_!timestamp!.log | Select-Object -Last 10"
) else (
    echo.
    echo ❌ CNN baseline training failed with error code !errorlevel!
    echo Check log: logs\cnn_training_!timestamp!.log
    echo.
    echo Error details:
    powershell "Get-Content logs\cnn_training_!timestamp!.log | Select-Object -Last 20"
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
echo   run_training.bat [TYPE] --data-dir PATH [OPTIONS]
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
echo   --batch-size SIZE        Training batch size (default: 16)
echo   --epochs NUM             Number of training epochs (default: 20)
echo   --learning-rate RATE     Learning rate (default: 0.001)
echo   --lr RATE                Alias for --learning-rate
echo   --time-steps NUM         SNN simulation time steps (default: 10)
echo   -T NUM                   Alias for --time-steps
echo   --mode MODE              SNN mode: train/cross_validation/comprehensive (default: train)
echo   --device DEVICE          Device: auto/cpu/cuda (default: auto)
echo   --no-timestamps          Disable timestamped folders
echo   --skip-inception         Skip InceptionV3 training (CNN only)
echo   --help, -h               Show this help message
echo.
echo EXAMPLES:
echo   # Train both SNN and CNN models
echo   run_training.bat both --data-dir C:\data\SDNET2018
echo.
echo   # Train only SNN with custom timesteps
echo   run_training.bat snn --data-dir C:\data\SDNET2018 --time-steps 20 --mode cross_validation
echo.
echo   # Train SNN with fewer timesteps for faster training
echo   run_training.bat snn --data-dir C:\data\SDNET2018 --time-steps 5 --epochs 10
echo.
echo   # Train CNN baselines with custom settings
echo   run_training.bat cnn --data-dir C:\data\SDNET2018 --batch-size 32 --epochs 10 --skip-inception
echo.
echo   # Quick training run without timestamps
echo   run_training.bat both --data-dir C:\data\SDNET2018 --epochs 5 --time-steps 4 --no-timestamps
echo.
goto end_script

:end_script
echo.
echo ================================================================================
echo                                TRAINING COMPLETE
echo ================================================================================
echo.
echo Summary files and training logs can be found in:
echo   - checkpoints\         (model checkpoints)
echo   - results\             (evaluation results)  
echo   - logs\                (training logs)
echo.
echo Completed at %time%
echo.
pause
exit /b 0