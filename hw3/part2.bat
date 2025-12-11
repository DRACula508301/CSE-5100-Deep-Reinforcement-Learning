@echo off
echo Activating virtual environment...
call .venv\Scripts\activate

echo [1/4] Running: cartpole_rtg_no_baseline
python run.py --env_name CartPole-v1 -n 100 -b 5000 -rtg --exp_name cartpole_rtg_no_baseline
echo [2/4] Running: cartpole_na_rtg_baseline
python run.py --env_name CartPole-v1 -n 100 -b 5000 -rtg -na --use_baseline --exp_name cartpole_na_rtg_baseline
echo [3/4] Running: cartpole_rtg_baseline_bgs3
python run.py --env_name CartPole-v1 -n 100 -b 5000 -rtg -na --use_baseline --exp_name cartpole_rtg_baseline_bgs3 -bgs 3
echo [4/4] Running: cartpole_rtg_baseline_blr5e-4
python run.py --env_name CartPole-v1 -n 100 -b 5000 -rtg -na --use_baseline --exp_name cartpole_rtg_baseline_blr5e-4 -blr 5e-4

echo All experiments finished.
pause