import shimmy # type: ignore
from shimmy import MeltingPotCompatibilityV0 # type: ignore
import torch # type: ignore
import supersuit as ss # type: ignore
import gymnasium as gym # type: ignore

from MoralRewardWrapper import MoralRewardWrapper
from training import train, parse_args
from utils import make_vec_env, SSWrapper

def main():
    print("Hello from final-project-5100!")
    # 1. Load Harvest substrate using Shimmy
    # Set render_mode to "rgb_array" for faster training
    env = MeltingPotCompatibilityV0(substrate_name="commons_harvest__open", render_mode="rgb_array")

    # 2. Wrap with MoralRewardWrapper for different moral reward types
    # util_env = MoralRewardWrapper(env, moral_reward_type='utilitarian')
    # imitation_env = MoralRewardWrapper(env, moral_reward_type='imitation', imitation_model=DummyImitationModel())
    selfish_env = MoralRewardWrapper(env, moral_reward_type='selfish')

    # 3. Apply supersuit wrappers
    selfish_env = SSWrapper(selfish_env)

    args = parse_args()
    
    selfish_env = make_vec_env(selfish_env, args)
    args.num_envs = selfish_env.num_envs

    # Recalculate batch sizes because num_envs has changed
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    # Train selfish agent here using selfish_env
    train(selfish_env, args)
    
if __name__ == "__main__":
    main()