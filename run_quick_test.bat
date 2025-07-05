@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8

echo ==========================================
echo QUICK TEST: Enhanced Features Validation
echo ==========================================
echo.
echo This script runs quick tests to validate:
echo - Enhanced augmentation functionality
echo - Threshold tuning integration
echo - Dataset type filtering
echo.

set DATA_DIR=C:\Users\anhnd\Desktop\TEST\crack_detection_snn\SDNET2018
set EPOCHS=2
set BATCH_SIZE=16

echo Data directory: %DATA_DIR%
echo Epochs: %EPOCHS% (quick test)
echo Batch size: %BATCH_SIZE% (small for speed)
echo.

REM Check if data directory exists
if not exist "%DATA_DIR%" (
    echo ERROR: Dataset directory not found: %DATA_DIR%
    echo Please update DATA_DIR in this script to point to your SDNET2018 dataset
    pause
    exit /b 1
)

echo Starting quick validation tests...
echo.

REM ==========================================
REM TEST 1: SNN with deck-only dataset
REM ==========================================
echo.
echo ==========================================
echo TEST 1: SNN - Deck dataset with enhanced features
echo ==========================================
echo.

python spiking_concrete.py ^
    --data-dir "%DATA_DIR%" ^
    --dataset-type deck ^
    --batch-size %BATCH_SIZE% ^
    --num-epochs %EPOCHS% ^
    --use-enhanced-augmentation ^
    --enable-threshold-tuning ^
    --threshold-metric f1

if errorlevel 1 (
    echo ERROR: SNN quick test failed
    pause
    exit /b 1
) else (
    echo SUCCESS: SNN quick test completed
)

echo.

REM ==========================================
REM TEST 2: CNN with pavement-only dataset
REM ==========================================
echo.
echo ==========================================
echo TEST 2: CNN - Pavement dataset with enhanced features
echo ==========================================
echo.

python baseline_concrete.py ^
    --data-dir "%DATA_DIR%" ^
    --dataset-type pavement ^
    --batch-size %BATCH_SIZE% ^
    --num-epochs %EPOCHS% ^
    --architecture resnet18 ^
    --use-enhanced-augmentation ^
    --enable-threshold-tuning ^
    --threshold-metric balanced_accuracy

if errorlevel 1 (
    echo ERROR: CNN quick test failed
    pause
    exit /b 1
) else (
    echo SUCCESS: CNN quick test completed
)

echo.
echo ==========================================
echo QUICK TESTS COMPLETED SUCCESSFULLY
echo ==========================================
echo.
echo Both enhanced augmentation and threshold tuning are working correctly!
echo Check the results/ directory for threshold tuning plots and metrics.
echo.
pause