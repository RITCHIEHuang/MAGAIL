#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/16 下午5:53
import unittest

import torch
import torch.nn.functional as F

from custom.MultiSoftMax import MultiSoftMax


class MultiSoftMaxTest(unittest.TestCase):

    def test_multi_softmax_head(self):
        """
        x[:2] softmax
        x[2: 2 + 3] softmax
        x[5: 5 + 3] softmax
        x[8: 8 + 2] softmax
        :return:
        """
        x = torch.randn((4, 10))
        sections = (2, 3, 3, 2)

        out = MultiSoftMax(0, 10, sections)(x)
        self.assertEqual(x.size(), out.size())
        self.assertTrue(torch.equal(out[..., :2], F.softmax(x[..., :2], dim=1)))
        self.assertTrue(torch.equal(out[..., 2:2 + 3], F.softmax(x[..., 2:2 + 3], dim=1)))
        self.assertTrue(torch.equal(out[..., 5:5 + 3], F.softmax(x[..., 5:5 + 3], dim=1)))
        self.assertTrue(torch.equal(out[..., 8:8 + 2], F.softmax(x[..., 8:8 + 2], dim=1)))

    def test_multi_softmax_with_tail(self):
        """
        x[0:2] normal
        x[2: 2 + 3] softmax
        x[5: 5 + 3] softmax
        x[8: 8 + 2] softmax
        :return:
        """
        x = torch.randn((4, 10))
        sections = (3, 3, 2)

        out = MultiSoftMax(2, 10, sections)(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.equal(out[..., :2], x[..., :2]))
        self.assertTrue(torch.equal(out[..., 2:2 + 3], F.softmax(x[..., 2:2 + 3], dim=1)))
        self.assertTrue(torch.equal(out[..., 5:5 + 3], F.softmax(x[..., 5:5 + 3], dim=1)))
        self.assertTrue(torch.equal(out[..., 8:8 + 2], F.softmax(x[..., 8:8 + 2], dim=1)))

    def test_multi_softmax_median(self):
        """
        x[0:2] normal
        x[2: 2 + 3] softmax
        x[5: 5 + 3] softmax
        x[-2:] normal
        :return:
        """
        x = torch.randn((4, 10))
        sections = (3, 3)

        out = MultiSoftMax(2, 8, sections)(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.equal(out[..., :2], x[..., :2]))
        self.assertTrue(torch.equal(out[..., 2:2 + 3], F.softmax(x[..., 2:2 + 3], dim=1)))
        self.assertTrue(torch.equal(out[..., 5:5 + 3], F.softmax(x[..., 5:5 + 3], dim=1)))
        self.assertTrue(torch.equal(out[..., -2:], x[..., -2:]))

    def test_multi_softmax_none(self):
        """
        softmax all features
        :return:
        """
        x = torch.randn((4, 10))
        out = MultiSoftMax(0, 10, [10])(x)
        self.assertTrue(torch.equal(out, F.softmax(x, dim=1)))


if __name__ == '__main__':
    unittest.main()
