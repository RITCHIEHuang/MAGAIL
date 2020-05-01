#!/usr/bin/env python
# Created at 2020/3/10
from unittest import TestCase

import torch

from models.mlp_discriminator import Discriminator


class TestDiscriminator(TestCase):
    def setUp(self) -> None:
        self.d = Discriminator(6, 5, drop_rate=0.5)
        print(self.d)

    def test_forward(self):
        res = self.d.forward(torch.rand((6, 6)), torch.rand((6, 5)))
        self.assertEqual(res.size(), torch.Size([6, 1]))
