## Setup the environment
```
# remove the content 'pytorch-cuda=11.7 -c pytorch -c nvidia' if you are a mac user or are not going to use GPU
conda install pytorch==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install 'gymnasium[classic_control]==0.27.1'
pip install matplotlib==3.7.1
pip install tensorboardX==2.6.4
```

## Complete the code

The files that you are going to implement are:

- `src/pg_agent.py`
- `src/policies.py`
- `src/critics.py`
- `src/utils.py`

See the [Assignment PDF](hw3.pdf) for more instructions.

## Submission

You should submit your code and the training logs, as well as your report on Canvas.