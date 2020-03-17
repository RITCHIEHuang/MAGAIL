#!/usr/bin/env python
# Created at 2020/3/10
from unittest import TestCase

import torch

from models.mlp_critic import Value


class TestValue(TestCase):
    def setUp(self) -> None:
        self.value = Value(6, drop_rate=0.5)
        print(self.value)

    def test_forward(self):
        res = self.value.forward(torch.rand((5, 6)))

        self.assertEqual(res.size(), torch.Size([5, 1]))
