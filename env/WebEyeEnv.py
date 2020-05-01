#!/usr/bin/env python
# Created at 2020/4/30

import gym
import torch
from gym import spaces

from data.ExpertDataSet import ExpertDataSet
from main import config_loader
from utils.torch_util import device


class WebEyeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self, config_path, model_path):
        self.config = config_loader(path=config_path)
        self.model = torch.load(model_path, map_location=device)  # load pre-trained policy (JointPolicy used here)
        self.observation_space = spaces.Dict({
            'discrete': spaces.MultiDiscrete((5, 2, 4, 3, 2, 9, 2, 32, 35, 7, 2, 21, 2, 3, 3)),
            'continuous': spaces.Box(-1, 1, (23,))
        })
        self.action_space = spaces.Box(-1, 1, (6,))

        dataset = ExpertDataSet(data_set_path=self.config["dataset_path"],
                                num_states=self.config["observation"]["space"],
                                num_actions=self.config["action"]["space"])
        self._init_state = dataset.state
        self.cur_step = 0
        self.max_step = self.config["max_step"]
        self.state = None
        self.reset()

    def step(self, action):
        self.state = self.model.get_next_state(torch.from_numpy(self.state).to(device), action).numpy()
        self.cur_step += 1
        done = (self.cur_step >= self.max_step)
        reward = self._calc_reward()
        return self.state, reward, done, {}

    def seed(self, seed=None):
        torch.manual_seed(seed)

    def _calc_reward(self):
        return self.state[:, self.config["reward_index"]]

    def reset(self, batch_size=1):
        self.state = (self._init_state[torch.randint(self._init_state.shape[0], size=(batch_size,))]).numpy()
        self.cur_step = 0
        return self.state

    def render(self, mode='human'):
        pass
