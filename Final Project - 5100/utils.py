import supersuit as ss
import gymnasium as gym

def SSWrapper(env):
    # Example Supersuit wrapper function
    env = ss.observation_lambda_v0(env, lambda obs, obs_space: obs['RGB'], lambda obs_space: obs_space['RGB'])
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    return env

def make_vec_env(env, args):
    """
    Applies supersuit and gym wrappers to convert a PettingZoo environment
    into a vectorized, Gymnasium-style environment.
    """
    env = ss.concat_vec_envs_v1(env, args.num_envs, num_cpus=0, base_class="gymnasium")
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    env.is_vector_env = True
    # env = gym.wrappers.vector.RecordEpisodeStatistics(env)
    # Video recording is handled inside the training script, so no need to wrap it here.
    assert isinstance(env.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    return env
