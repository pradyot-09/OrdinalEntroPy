import numpy as np
import unittest

# from ordinal_entropy import (PE, WPE, RPE,
#                      RWPE, DE, RDE, RWDE)

from OrdinalEntroPy import (PE, WPE, RPE, RWPE, DE, RDE, RWDE)

np.random.seed(1234567)
RANDOM_TS = np.random.rand(3000)
RANDOM_TS_LONG = np.random.rand(6000)
SF_TS = 100
BANDT_PERM = [4, 7, 9, 10, 6, 11, 3]
x_de = [9, 8, 1, 12, 5, -3, 1.5, 8.01, 2.99, 4, -1, 10]
PURE_SINE = np.sin(2 * np.pi * 1 * np.arange(3000) / 100)


class TestEntropy(unittest.TestCase):
    def test_PE(self):
        self.assertEqual(np.round(PE(RANDOM_TS, order=3, delay=1, normalize=True), 1),1.0)
        # Compare with Bandt and Pompe 2002
        self.assertEqual(np.round(PE(BANDT_PERM, order=2,normalize=False), 3), 0.918)
        self.assertEqual(np.round(PE(BANDT_PERM, order=3,normalize=False), 3), 1.522)
        # Error
        with self.assertRaises(ValueError):
            PE(BANDT_PERM, order=4, delay=3)
        with self.assertRaises(ValueError):
            PE(BANDT_PERM, order=3, delay=0.5)
        with self.assertRaises(ValueError):
            PE(BANDT_PERM, order=1, delay=1)

    def test_WPE(self):
        WPE(BANDT_PERM,order=3,delay=1,normalize=True)
        with self.assertRaises(ValueError):
            WPE(BANDT_PERM, order=4, delay=3)
        with self.assertRaises(ValueError):
            WPE(BANDT_PERM, order=3, delay=0.5)
        with self.assertRaises(ValueError):
            WPE(BANDT_PERM, order=1, delay=1)
        
    def test_DE(self):
        DE(x_de, order=2,classes=3,delay=2,normalize=True)
        self.assertEqual(np.round(DE(x_de, order=2,classes=3,normalize=True), 2), 0.84)
        # Error
        with self.assertRaises(ValueError):
            DE(x_de, order=3,classes=1)


    def test_RPE(self):
        RPE(RANDOM_TS, order=2,delay=2,normalize=True)
        RPE(RANDOM_TS, order=3,normalize=True)
        self.assertEqual(np.round(RPE(BANDT_PERM, order=3), 3), 0.232)

    def test_RWPE(self):
        RWPE(RANDOM_TS, order=2,delay=2,normalize=True)
        RWPE(RANDOM_TS, order=3,normalize=True)

    def test_RDE(self):
        RDE(RANDOM_TS, order=2,classes=3,delay=2,normalize=True)
        RDE(RANDOM_TS, order=3,classes=3,normalize=True)
        self.assertEqual(np.round(RDE(BANDT_PERM, order=2,classes=3), 3), 0.125)
        with self.assertRaises(ValueError):
            RDE(BANDT_PERM, order=4,classes=3, delay=3)
        with self.assertRaises(ValueError):
            RDE(BANDT_PERM, order=3, classes=3, delay=0.5)
        with self.assertRaises(ValueError):
            RDE(BANDT_PERM, order=1, classes=3, delay=1)
    
    def test_RWDE(self):
        RDE(RANDOM_TS, order=2,classes=3,delay=2,normalize=True)
        RDE(RANDOM_TS, order=3,classes=3,normalize=True)
        self.assertEqual(np.round(RWDE(BANDT_PERM, order=2,classes=3), 3), 0.294)
