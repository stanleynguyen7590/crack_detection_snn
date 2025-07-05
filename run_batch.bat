@echo off

:: Fix charmap codec issues
chcp 65001 > nul
set PYTHONIOENCODING=utf-8

echo Running crack detection experiments...

python spiking_concrete.py --data-dir ./SDNET2018/ --class-scheme 3class --use-weighted-sampling --num-epochs 50 --time-steps 10 --batch-size 16
python spiking_concrete.py --data-dir ./SDNET2018/ --class-scheme binary --use-weighted-sampling --num-epochs 50 --time-steps 10 --batch-size 16
python spiking_concrete.py --data-dir ./SDNET2018/ --class-scheme 6class --use-weighted-sampling --num-epochs 50 --time-steps 10 --batch-size 16

python baseline_concrete.py --data-dir ./SDNET2018/ --class-scheme 3class --use-weighted-sampling --num-epochs 50 --batch-size 16
python baseline_concrete.py --data-dir ./SDNET2018/ --class-scheme binary --use-weighted-sampling --num-epochs 50 --batch-size 16
python baseline_concrete.py --data-dir ./SDNET2018/ --class-scheme 6class --use-weighted-sampling --num-epochs 50 --batch-size 16

echo All experiments completed!
pause