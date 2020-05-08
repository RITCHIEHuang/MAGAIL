#!/usr/bin/env python
# Created at 2020/3/10
from unittest import TestCase

import torch

from models.mlp_critic import Value


class TestValue(TestCase):
    def setUp(self) -> None:
        self.value = Value(6, drop_rate=0.5)
        self.value2 = Value(11, drop_rate=0.5)
        print(self.value)

    def test_forward(self):
        res = self.value.forward(torch.rand((5, 6)))

        self.assertEqual(res.size(), torch.Size([5, 1]))

    def test_multi_forward(self):
        x1 = torch.rand((5, 6))
        x2 = torch.rand((5, 5))

        res = self.value2.forward(x1, x2)

        print(res)
