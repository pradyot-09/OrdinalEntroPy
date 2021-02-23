import numpy as np
import unittest

from ordinal_entropy import (perm_entropy, spectral_entropy, svd_entropy,
                     sample_entropy, app_entropy, lziv_complexity)

np.random.seed(1234567)
RANDOM_TS = np.random.rand(3000)
RANDOM_TS_LONG = np.random.rand(6000)
SF_TS = 100
BANDT_PERM = [4, 7, 9, 10, 6, 11, 3]
PURE_SINE = np.sin(2 * np.pi * 1 * np.arange(3000) / 100)


class TestEntropy(unittest.TestCase):
    def test_perm_entropy(self):
    self.assertEqual(np.round(perm_entropy(RANDOM_TS, order=3,
                                            delay=1, normalize=True), 1),
                        1.0)
    # Compare with Bandt and Pompe 2002
    self.assertEqual(np.round(perm_entropy(BANDT_PERM, order=2), 3), 0.918)
    self.assertEqual(np.round(perm_entropy(BANDT_PERM, order=3), 3), 1.522)
    # Error
    with self.assertRaises(ValueError):
        perm_entropy(BANDT_PERM, order=4, delay=3)
    with self.assertRaises(ValueError):
        perm_entropy(BANDT_PERM, order=3, delay=0.5)
    with self.assertRaises(ValueError):
        perm_entropy(BANDT_PERM, order=1, delay=1)