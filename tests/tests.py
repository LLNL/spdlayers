# Copyright 2021, Lawrence Livermore National Security, LLC and spdlayer
# contributors
# SPDX-License-Identifier: MIT

import unittest
import numpy as np
import torch
from spdlayers import Eigen, Cholesky
from spdlayers import in_shape_from

positive_funs = ['Abs', 'Square', 'Softplus', 'ReLU', 'ReLU6', '4', 'Exp']
batch_size = 1000


class TestSPD(unittest.TestCase):

    def test_input_shape(self):
        in_shape = in_shape_from(6)
        self.assertTrue(in_shape == 21)
        self.assertIsInstance(in_shape, int)

    def test_Eigen_anisotropic(self):
        x = torch.rand(batch_size, 21).double()
        for pos in positive_funs:
            myEigen = Eigen(output_shape=6, positive=pos).double()
            out = myEigen(x)
            u = torch.real(torch.linalg.eigvals(out))
            min_eig_val = torch.min(u).item()
            self.assertTrue(min_eig_val > 0.0)

    def test_Cholesky_anisotropic(self):
        x = torch.rand(batch_size, 21).double()
        for pos in positive_funs:
            myCholesky = Cholesky(output_shape=6, positive=pos).double()
            out = myCholesky(x)
            u = torch.real(torch.linalg.eigvals(out))
            min_eig_val = torch.min(u).item()
            if np.isclose(min_eig_val, 0.0, rtol=1e-10, atol=1e-10):
                min_eig_val = 0.0
            self.assertTrue(min_eig_val >= 0.0)

    def test_Eigen_orthotropic(self):
        x = torch.rand(batch_size, 9).double()
        for pos in positive_funs:
            myEigen = Eigen(output_shape=6,
                            symmetry='orthotropic',
                            positive=pos).double()
            out = myEigen(x)
            u = torch.real(torch.linalg.eigvals(out))
            min_eig_val = torch.min(u).item()
            self.assertTrue(min_eig_val > 0.0)

    def test_Cholesky_orthotropic(self):
        x = torch.rand(batch_size, 9).double()
        for pos in positive_funs:
            myCholesky = Cholesky(output_shape=6,
                                  symmetry='orthotropic',
                                  positive=pos).double()
            out = myCholesky(x)
            u = torch.real(torch.linalg.eigvals(out))
            min_eig_val = torch.min(u).item()
            if np.isclose(min_eig_val, 0.0, rtol=1e-10, atol=1e-10):
                min_eig_val = 0.0
            self.assertTrue(min_eig_val >= 0.0)

    def test_value_errors_eig(self):
        try:
            _ = Eigen(symmetry='bob')
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        try:
            _ = Eigen(output_shape=7, symmetry='orthotropic')
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        try:
            _ = Eigen(positive='hello')
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_value_errors_chol(self):
        try:
            _ = Cholesky(symmetry='bob')
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        try:
            _ = Cholesky(output_shape=7, symmetry='orthotropic')
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        try:
            _ = Cholesky(positive='hello')
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
