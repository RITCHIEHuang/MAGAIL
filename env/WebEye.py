#!/usr/bin/env python
# Created at 2020/4/30

import gym
import torch
from gym import spaces

from data.ExpertDataSet import ExpertDataSet
from utils.torch_utils import device


class WebEyeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self, config, model_path, max_length=1):
        self.config = config
        self.model = torch.load(model_path, map_location=device)  # load pre-trained policy
        self.observation_space = spaces.Dict({
            'discrete_space': spaces.MultiDiscrete((5, 2, 4, 3, 2, 9, 2, 32, 35, 7, 2, 21, 2, 3, 3)),
            'continuous_space': spaces.Box(-1, 1, (23,))
        })
        self.action_space = spaces.Dict({
            'discrete_space': spaces.MultiDiscrete((0,)),
            'continuous_space': spaces.Box(-1, 1, (6,))
        })

        dataset = ExpertDataSet(data_set_path=config["dataset_path"],
                                num_states=config["observation"]["space"],
                                num_actions=config["action"]["space"])
        self._init_state = dataset.state
        self.max_length = max_length
        self.length = 0
        self.state = None
        self.reset()

    def step(self, action):
        self.state, _ = self.model.env_policy.get_next_state(self.state, action)
        self.length += 1
        done = (self.length >= self.max_length)
        reward = self._calc_reward()
        return self.state, reward, done, {}

    def seed(self, seed=None):
        torch.manual_seed(seed)

    def _calc_reward(self):
        return self.state[:, self.config["reward_index"]].to(device)

    def reset(self, batch_size=1):
        self.state = self._init_state[torch.randint(self._init_state.shape[0], size=(batch_size,))] \
            .to(device)
        self.length = 0
        return self.state

    def render(self, mode='human'):
        pass
