import numpy as np # type: ignore
import supersuit as ss # type: ignore
from pettingzoo.utils.wrappers import BaseParallelWrapper

class MoralRewardWrapper(BaseParallelWrapper):
    # Takes in an PettingZoo environment and a moral reward type (utilitarian or imitation)
    # Overrides step function to inject intrinsic moral rewards (Utilitarian/Imitation) before passing them to the agent
 
    def __init__(self, env, moral_reward_type='utilitarian', imitation_model=None):
        super().__init__(env)
        self.moral_reward_type = moral_reward_type
        self.imitation_model = imitation_model
    
    def __getattr__(self, name):
        """Forward attribute access to the wrapped environment, ignoring private attributes."""
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, extrinsic_rewards, terms, truncs, infos = self.env.step(action)
        
        intrinsic_rewards = {agent: 0.0 for agent in self.agents}
        if self.moral_reward_type == 'utilitarian':
            # Example: Reward based on collective apple collection
            intrinsic_rewards = self.calculate_utilitarian_reward(extrinsic_rewards)
            return obs, intrinsic_rewards, terms, truncs, infos
        elif self.moral_reward_type == 'imitation' and self.imitation_model is not None:
            # Example: Reward based on similarity to expert behavior
            intrinsic_rewards = self.calculate_imitation_reward(obs)
        elif self.moral_reward_type == 'selfish':
            intrinsic_rewards = extrinsic_rewards
        
        # total_reward = extrinsic_reward + intrinsic_reward
        return obs, intrinsic_rewards, terms, truncs, infos

    def close(self):
        self.env.close()

    def calculate_utilitarian_reward(self, extrinsic_rewards: dict) -> dict:
        # Reward is average reward of all agents
        total = sum(extrinsic_rewards.values())
        avg_reward = total / len(extrinsic_rewards) if extrinsic_rewards else 0
        return {agent: avg_reward for agent in extrinsic_rewards.keys()}
    
    def calculate_imitation_reward(self, obs) -> dict:
        # Placeholder for imitation reward calculation
        # In practice, this would compare agent actions to those predicted by the imitation model
        imitation_rewards = {}
        for agent in self.agents:
            predicted_action = self.imitation_model.predict(obs[agent])
            actual_action = self.env.last(agent)[0] # Get last action for agent
            imitation_rewards[agent] = 1.0 if predicted_action == actual_action else 0.0
        return imitation_rewards
