import unittest

import torch

from two_step_kpp.models import DPResNet, MFMLP


class ModelsTest(unittest.TestCase):
    def test_dp_resnet_initial_identity(self):
        model = DPResNet(dim=2)
        x = torch.randn(16, 2)
        sigma = torch.randn(16)
        out = model(x, sigma)
        self.assertTrue(torch.allclose(out, x, atol=1e-5))

    def test_mfmlp_jvp_shape(self):
        model = MFMLP(dim=3)
        z = torch.randn(8, 3)
        r = torch.rand(8)
        t = torch.rand(8)
        sigma = torch.randn(8)
        v = torch.randn(8, 3)
        u, u_dot = model.jvp_u(z, r, t, sigma, v)
        self.assertEqual(u.shape, (8, 3))
        self.assertEqual(u_dot.shape, (8, 3))


if __name__ == "__main__":
    unittest.main()
