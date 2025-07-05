@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8

echo ==========================================
echo COMPREHENSIVE CRACK DETECTION EXPERIMENTS
echo ==========================================
echo.
echo This script runs systematic experiments with:
echo - 4 dataset types: all, deck, pavement, wall
echo - Both SNN and CNN models
echo - Enhanced augmentation and threshold tuning
echo - Binary classification focus
echo.

set DATA_DIR=C:\Users\anhnd\Desktop\TEST\crack_detection_snn\SDNET2018
set EPOCHS=20
set BATCH_SIZE=16

echo Data directory: %DATA_DIR%
echo Epochs: %EPOCHS%
echo Batch size: %BATCH_SIZE%
echo.

REM Check if data directory exists
if not exist "%DATA_DIR%" (
    echo ERROR: Dataset directory not found: %DATA_DIR%
    echo Please update DATA_DIR in this script to point to your SDNET2018 dataset
    pause
    exit /b 1
)

echo Starting comprehensive experiments...
echo.

REM ==========================================
REM PART 1: SNN EXPERIMENTS
REM ==========================================
echo.
echo ==========================================
echo PART 1: SPIKING NEURAL NETWORK EXPERIMENTS
echo ==========================================
echo.

set DATASET_TYPES=all deck pavement wall

for %%d in (%DATASET_TYPES%) do (
    echo.
    echo ##########################################
    echo SNN Training: %%d dataset
    echo ##########################################
    echo.
    
    python spiking_concrete.py ^
        --data-dir "%DATA_DIR%" ^
        --dataset-type %%d ^
        --batch-size %BATCH_SIZE% ^
        --num-epochs %EPOCHS% ^
        --use-enhanced-augmentation ^
        --enable-threshold-tuning ^
        --threshold-metric f1
    
    if errorlevel 1 (
        echo ERROR: SNN training failed for %%d dataset
        echo Continuing with next experiment...
    ) else (
        echo SUCCESS: SNN training completed for %%d dataset
    )
    echo.
)

REM ==========================================
REM PART 2: CNN EXPERIMENTS
REM ==========================================
echo.
echo ==========================================
echo PART 2: CNN BASELINE EXPERIMENTS
echo ==========================================
echo.

for %%d in (%DATASET_TYPES%) do (
    echo.
    echo ##########################################
    echo CNN Training: %%d dataset
    echo ##########################################
    echo.
    
    python baseline_concrete.py ^
        --data-dir "%DATA_DIR%" ^
        --dataset-type %%d ^
        --batch-size 16 ^
        --num-epochs 15 ^
        --architecture all ^
        --use-enhanced-augmentation ^
        --enable-threshold-tuning ^
        --threshold-metric f1
    
    if errorlevel 1 (
        echo ERROR: CNN training failed for %%d dataset
        echo Continuing with next experiment...
    ) else (
        echo SUCCESS: CNN training completed for %%d dataset
    )
    echo.
)

echo.
echo ==========================================
echo ALL EXPERIMENTS COMPLETED
echo ==========================================
echo.
echo Check the following directories for results:
echo - checkpoints/ : Trained model files
echo - results/     : Evaluation metrics and plots
echo.
echo Summary files (*_summary.json) are created in base directories
echo pointing to the timestamped experiment folders.
echo.
pause