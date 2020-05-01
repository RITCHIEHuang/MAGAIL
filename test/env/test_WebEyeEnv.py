#!/usr/bin/env python
import unittest
from unittest import TestCase

# Created at 2020/5/1
import torch

from env.WebEyeEnv import WebEyeEnv
from utils.torch_util import FLOAT, device


class TestWebEyeEnv(TestCase):
    def setUp(self) -> None:
        self.env = WebEyeEnv(config_path="../../config/config_webeye_env.yml",
                             model_path="../../model_pkl/MAGAIL_Train_2020-05-01_18:09:33_JointPolicy.pt",
                             )

    def test_run(self):
        print(self.env.observation_space)
        print(self.env.action_space)
        state = self.env.reset()
        for i in range(10):
            action = self.env.action_space.sample()
            print("state: ", state)
            print("action: ", action)

            state, reward, done,  _ = self.env.step(torch.from_numpy(action).unsqueeze(0).to(device))


if __name__ == '__main__':
    unittest.main()
