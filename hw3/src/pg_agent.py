from typing import Optional, Sequence
import numpy as np
import torch

from src.policies import MLPPolicyPG
from src.critics import ValueCritic
import src.pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        obs = np.concatenate(obs)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        terminals = np.concatenate(terminals)
        q_values = np.concatenate(q_values)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # update the PG actor/policy network once using the advantages
        info: dict = self.actor.update(obs, actions, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # perform `self.baseline_gradient_steps` updates to the critic/baseline network
            critic_info: dict = None
            ############################
            # YOUR IMPLEMENTATION HERE #

            ############################

            info.update(critic_info)

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # TODO: use the helper function self._discounted_return to calculate the Q-values
            q_values = None
            ############################
            # YOUR IMPLEMENTATION HERE #
            q_values = [self._discounted_return(rew.tolist()) for rew in rewards]
            ############################

        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
            q_values = None

            ############################
            # YOUR IMPLEMENTATION HERE #
            q_values = [self._discounted_reward_to_go(rew.tolist()) for rew in rewards]
            ############################

        return [np.array(qv) for qv in q_values]

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        advantages: np.ndarray = None
        if self.critic is None:   
            advantages = q_values.copy()
        else:
            # run the critic and use it as a baseline to compute values and advantages
            values = None
            advantages = None
            ############################
            # YOUR IMPLEMENTATION HERE #
            
            ############################
            assert values.shape == q_values.shape

        # normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            
            ############################
            # YOUR IMPLEMENTATION HERE #
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            ############################

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!

        self.gamma
        """

        ############################
        # YOUR IMPLEMENTATION HERE #
        # return empty array for empty trajectory
        if len(rewards) == 0:
            return []
        discounted_sum = 0
        for t, r in enumerate(rewards):
            discounted_sum += (self.gamma ** float(t)) * r
        returns = [discounted_sum] * len(rewards)
        ############################
        return returns


    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.

        self.gamma
        """

        ############################
        # YOUR IMPLEMENTATION HERE #
        if len(rewards) == 0:
            return []
        n = len(rewards)
        returns = [0] * n
        for t in range(n):
            discounted_sum = 0
            for t_prime in range(t, n):
                discounted_sum += (self.gamma ** float(t_prime - t)) * rewards[t_prime]
            returns[t] = discounted_sum
        return returns
        ############################