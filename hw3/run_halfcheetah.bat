@echo off
echo Activating virtual environment...
call .venv\Scripts\activate

echo [1/3] Running: halfcheetah_rtg_no_baseline
python run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --exp_name halfcheetah_rtg_no_baseline --discount 0.99
echo [2/3] Running: halfcheetah_na_rtg_baseline
python run.py --env_name HalfCheetah -n 100 -b 5000 -rtg -na --use_baseline --exp_name halfcheetah_rtg_baseline --discount 0.99 -l 2 -s 32
echo [3/3] Running: halfcheetah_rtg_baseline_blr1e-3_lr3e-4
python run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg -na --use_baseline --exp_name halfcheetah_rtg_baseline_blr1e-3_lr3e-4 --discount 0.99 -l 2 -s 32 -blr 1e-3 -lr 3e-4

echo All experiments finished.
pause