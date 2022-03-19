#!/usr/bin/env python
# Created at 2020/3/12
import torch
import torch.nn as nn

from magail.Memory import Memory
from models.mlp_actor import Actor


class JointPolicy(nn.Module):
    """
    Joint Policy include:
    agent policy: (agent_state,) -> agent_action
    env policy: (agent_state, agent_action) -> agent_next_state
    """

    def __init__(self, initial_state, config=None):
        super(JointPolicy, self).__init__()
        self.config = config
        self.trajectory_length = config["trajectory_length"]
        self.agent_policy = Actor(num_states=self.config["agent"]["num_states"],
                                  num_actions=self.config["agent"]["num_actions"],
                                  num_discrete_actions=self.config["agent"]["num_discrete_actions"],
                                  discrete_actions_sections=self.config["agent"]["discrete_actions_sections"],
                                  action_log_std=self.config["agent"]["action_log_std"],
                                  use_multivariate_distribution=self.config["agent"]["use_multivariate_distribution"],
                                  num_hiddens=self.config["agent"]["num_hiddens"],
                                  drop_rate=self.config["agent"]["drop_rate"],
                                  activation=self.config["agent"]["activation"])

        self.env_policy = Actor(num_states=self.config["env"]["num_states"],
                                num_actions=self.config["env"]["num_actions"],
                                num_discrete_actions=self.config["env"]["num_discrete_actions"],
                                discrete_actions_sections=self.config["env"]["discrete_actions_sections"],
                                action_log_std=self.config["env"]["action_log_std"],
                                use_multivariate_distribution=self.config["env"]["use_multivariate_distribution"],
                                num_hiddens=self.config["env"]["num_hiddens"],
                                drop_rate=self.config["env"]["drop_rate"],
                                activation=self.config["env"]["activation"])

        # Joint policy generate trajectories sampling initial state from expert data
        self.initial_agent_state = initial_state

    def collect_samples(self, batch_size):
        """
        generate trajectories following current policy
        accelerate by parallel the process
        :param batch_size:
        :return:
        """
        memory = Memory()
        parallelize_size = (batch_size + self.trajectory_length - 1) // self.trajectory_length
        agent_state = self.initial_agent_state[torch.randint(self.initial_agent_state.shape[0], (
            parallelize_size,))]  # agent_state [parallelize_size, num_states]
        for i in range(1, self.trajectory_length + 1):
            with torch.no_grad():
                agent_action, agent_action_log_prob = self.agent_policy.get_action_log_prob(
                    agent_state if len(agent_state.shape) > 1 else agent_state.unsqueeze(-1))
                # agent_action [parallelize_size, num_actions], agent_action_log_prob [parallelize_size, 1]
                env_state = torch.cat([agent_state, agent_action],
                                      dim=-1)  # env_state [parallelize_size, num_states + num_actions]
                env_action, env_action_log_prob = self.env_policy.get_action_log_prob(
                    env_state if len(env_state.shape) > 1 else env_state.unsqueeze(
                        -1))  # env_action [parallelize_size, num_states], env_action_log_prob [parallelize_size, 1]

            assert agent_action_log_prob.shape == env_action_log_prob.shape, "Expected agent_policy log_prob and env_" \
                                                                             "policy log_prob with same size!!!"

            mask = torch.zeros_like(env_action_log_prob) if i % self.trajectory_length == 0 else torch.ones_like(
                env_action_log_prob)

            memory.push(agent_state, agent_action, env_action, agent_action_log_prob + env_action_log_prob, mask)

        return memory.sample()

    def get_log_prob(self, states, actions, next_states):
        agent_action_log_prob = self.agent_policy.get_log_prob(states, actions)
        env_states = torch.cat([states, actions], dim=1)
        env_action_log_prob = self.env_policy.get_log_prob(env_states, next_states)

        return agent_action_log_prob + env_action_log_prob

    def get_next_state(self, states, actions):
        state_actions = torch.cat([states, actions], dim=-1)
        next_state, _ = self.env_policy.get_action_log_prob(state_actions)
        return next_state
