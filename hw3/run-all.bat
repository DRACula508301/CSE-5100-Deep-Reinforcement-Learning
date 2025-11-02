@echo off
echo Activating virtual environment...
call .venv\Scripts\activate

echo [1/6] Running: cartpole_na
python run.py --env_name CartPole-v1 -n 200 -b 1000 -na --exp_name cartpole_na

echo [2/6] Running: cartpole_rtg_na
python run.py --env_name CartPole-v1 -n 200 -b 1000 -rtg -na --exp_name cartpole_rtg_na

echo [3/6] Running: cartpole_lb
python run.py --env_name CartPole-v1 -n 200 -b 4000 --exp_name cartpole_lb

echo [4/6] Running: cartpole_lb_rtg
python run.py --env_name CartPole-v1 -n 200 -b 4000 -rtg --exp_name cartpole_lb_rtg

echo [5/6] Running: cartpole_lb_na
python run.py --env_name CartPole-v1 -n 200 -b 4000 -na --exp_name cartpole_lb_na

echo [6/6] Running: cartpole_lb_rtg_na
python run.py --env_name CartPole-v1 -n 200 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na

echo All experiments finished.
pause