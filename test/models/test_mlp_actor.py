#!/usr/bin/env python
# Created at 2020/3/10
from unittest import TestCase

import torch

from models.mlp_actor import Actor


class TestActor(TestCase):
    def setUp(self) -> None:
        self.policy = Actor(3, 5)
        print(self.policy)

    def test_forward(self):
        res = self.policy.forward(torch.rand((6, 3)))
        self.assertEqual(res.size(), torch.Size([6, 3]))

    def test_get_action_log_prob(self):
        self.fail()

    def test_get_log_prob(self):
        self.fail()

    def test_get_entropy(self):
        self.fail()

    def test_get_kl(self):
        self.fail()
