@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8

echo ==========================================
echo THRESHOLD TUNING COMPARISON STUDY
echo ==========================================
echo.
echo This script compares different threshold optimization metrics:
echo - F1-score optimization
echo - Balanced accuracy optimization  
echo - Matthews Correlation Coefficient (MCC) optimization
echo.
echo Using wall dataset for focused comparison
echo.

set DATA_DIR=C:\Users\anhnd\Desktop\TEST\crack_detection_snn\SDNET2018
set EPOCHS=15
set BATCH_SIZE=8

echo Data directory: %DATA_DIR%
echo Epochs: %EPOCHS%
echo Batch size: %BATCH_SIZE%
echo Dataset: wall (most imbalanced for threshold tuning evaluation)
echo.

REM Check if data directory exists
if not exist "%DATA_DIR%" (
    echo ERROR: Dataset directory not found: %DATA_DIR%
    echo Please update DATA_DIR in this script to point to your SDNET2018 dataset
    pause
    exit /b 1
)

echo Starting threshold optimization comparison...
echo.

REM ==========================================
REM COMPARISON 1: F1-Score Optimization
REM ==========================================
echo.
echo ==========================================
echo EXPERIMENT 1: F1-Score Optimization
echo ==========================================
echo.

python spiking_concrete.py ^
    --data-dir "%DATA_DIR%" ^
    --dataset-type wall ^
    --batch-size %BATCH_SIZE% ^
    --num-epochs %EPOCHS% ^
    --use-enhanced-augmentation ^
    --enable-threshold-tuning ^
    --threshold-metric f1

if errorlevel 1 (
    echo ERROR: F1 optimization experiment failed
    echo Continuing with next experiment...
) else (
    echo SUCCESS: F1 optimization completed
)

echo.

REM ==========================================
REM COMPARISON 2: Balanced Accuracy Optimization
REM ==========================================
echo.
echo ==========================================
echo EXPERIMENT 2: Balanced Accuracy Optimization
echo ==========================================
echo.

python spiking_concrete.py ^
    --data-dir "%DATA_DIR%" ^
    --dataset-type wall ^
    --batch-size %BATCH_SIZE% ^
    --num-epochs %EPOCHS% ^
    --use-enhanced-augmentation ^
    --enable-threshold-tuning ^
    --threshold-metric balanced_accuracy

if errorlevel 1 (
    echo ERROR: Balanced accuracy optimization experiment failed
    echo Continuing with next experiment...
) else (
    echo SUCCESS: Balanced accuracy optimization completed
)

echo.

REM ==========================================
REM COMPARISON 3: MCC Optimization
REM ==========================================
echo.
echo ==========================================
echo EXPERIMENT 3: MCC Optimization
echo ==========================================
echo.

python spiking_concrete.py ^
    --data-dir "%DATA_DIR%" ^
    --dataset-type wall ^
    --batch-size %BATCH_SIZE% ^
    --num-epochs %EPOCHS% ^
    --use-enhanced-augmentation ^
    --enable-threshold-tuning ^
    --threshold-metric mcc

if errorlevel 1 (
    echo ERROR: MCC optimization experiment failed
    echo Continuing with next experiment...
) else (
    echo SUCCESS: MCC optimization completed
)

echo.

REM ==========================================
REM COMPARISON 4: CNN Baseline for Comparison
REM ==========================================
echo.
echo ==========================================
echo EXPERIMENT 4: CNN Baseline with F1 Optimization
echo ==========================================
echo.

python baseline_concrete.py ^
    --data-dir "%DATA_DIR%" ^
    --dataset-type wall ^
    --batch-size 16 ^
    --num-epochs %EPOCHS% ^
    --architecture resnet18 ^
    --use-enhanced-augmentation ^
    --enable-threshold-tuning ^
    --threshold-metric f1

if errorlevel 1 (
    echo ERROR: CNN baseline experiment failed
) else (
    echo SUCCESS: CNN baseline completed
)

echo.
echo ==========================================
echo THRESHOLD COMPARISON STUDY COMPLETED
echo ==========================================
echo.
echo Compare the results in the results/ directory:
echo - Look for *threshold_tuning*.png plots
echo - Check *threshold_tuning_results.json files
echo - Compare optimization metric improvements across experiments
echo.
echo Key files to examine:
echo 1. snn_*_threshold_tuning_*.png - Visualization plots
echo 2. snn_*_threshold_tuning_results.json - Detailed metrics
echo 3. Training results JSON files for overall performance
echo.
pause