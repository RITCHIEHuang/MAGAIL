#!/usr/bin/env python
# Created at 2020/3/12
import torch
import torch.nn as nn

from algos.Memory import Memory
from models.mlp_actor_v2 import Actor


class JointPolicy(nn.Module):
    """
    Joint Policy include:
    user policy: (user_state,) -> user_action
    env policy: (user_state, user_action) -> user_next_state
    """

    def __init__(self, initial_state, config=None):
        super(JointPolicy, self).__init__()
        self.config = config
        self.trajectory_length = config["trajectory_length"]
        self.user_policy = Actor(num_states=self.config["user"]["num_states"],
                                 num_actions=self.config["user"]["num_actions"],
                                 num_discrete_actions=self.config["user"]["num_discrete_actions"],
                                 discrete_actions_sections=self.config["user"]["discrete_actions_sections"],
                                 action_log_std=self.config["user"]["action_log_std"],
                                 use_multivariate_distribution=self.config["user"]["use_multivariate_distribution"],
                                 num_hiddens=self.config["user"]["num_hiddens"],
                                 drop_rate=self.config["user"]["drop_rate"])

        self.env_policy = Actor(num_states=self.config["env"]["num_states"],
                                num_actions=self.config["env"]["num_actions"],
                                num_discrete_actions=self.config["env"]["num_discrete_actions"],
                                discrete_actions_sections=self.config["env"]["discrete_actions_sections"],
                                action_log_std=self.config["env"]["action_log_std"],
                                use_multivariate_distribution=self.config["env"]["use_multivariate_distribution"],
                                num_hiddens=self.config["env"]["num_hiddens"],
                                drop_rate=self.config["env"]["drop_rate"])

        # Joint policy generate trajectories sampling initial state from expert data
        self.initial_user_state = initial_state

    def collect_samples(self, batch_size):
        """
        generate trajectories following current policy
        accelerate by parallel the process
        :param batch_size:
        :return:
        """
        memory = Memory()
        parallelize_size = (batch_size + self.trajectory_length - 1) // self.trajectory_length
        user_state = self.initial_user_state[torch.randint(self.initial_user_state.shape[0], (
            parallelize_size,))]  # user_state [parallelize_size, num_states]
        for i in range(1, self.trajectory_length + 1):
            with torch.no_grad():
                user_action, user_action_log_prob = self.user_policy.get_action_log_prob(
                    user_state if len(user_state.shape) > 1 else user_state.unsqueeze(
                        -1))  # user_action [parallelize_size, num_actions], user_action_log_prob [parallelize_size, 1]
                env_state = torch.cat([user_state, user_action],
                                      dim=-1)  # env_state [parallelize_size, num_states + num_actions]
                env_action, env_action_log_prob = self.env_policy.get_action_log_prob(
                    env_state if len(env_state.shape) > 1 else env_state.unsqueeze(
                        -1))  # env_action [parallelize_size, num_states], env_action_log_prob [parallelize_size, 1]

            assert user_action_log_prob.shape == env_action_log_prob.shape, "Expected user_policy log_prob and env_" \
                                                                            "policy log_prob with same size!!!"

            mask = torch.ones_like(env_action_log_prob) if i % self.trajectory_length == 0 else torch.zeros_like(
                env_action_log_prob)

            memory.push(user_state, user_action, env_action, user_action_log_prob + env_action_log_prob, mask)

            # # to re-use original gae code
            # for u_s, u_a, e_a, u_a_p, e_a_p, m in zip(user_state, user_action, env_action, user_action_log_prob,
            #                                           env_action_log_prob, mask):
            #     memory.push(u_s, u_a, e_a, u_a_p + e_a_p, m)

        return memory.sample()

    def get_log_prob(self, states, actions, next_states):
        user_action_log_prob = self.user_policy.get_log_prob(states, actions)
        env_states = torch.cat([states, actions], dim=1)
        env_action_log_prob = self.env_policy.get_log_prob(env_states, next_states)

        return user_action_log_prob + env_action_log_prob

    def get_next_state(self, states, actions):
        state_actions = torch.cat([states, actions], dim=-1)
        next_state, _ = self.env_policy.get_action_log_prob(state_actions)
        return next_state
