# Copyright 2021, Lawrence Livermore National Security, LLC and spdlayer
# contributors
# SPDX-License-Identifier: MIT

import unittest
import numpy as np
import torch
import torch.nn as nn
from spdlayers import Eigen, Cholesky
from spdlayers import in_shape_from

positive_funs = ['Abs', 'Square', 'Softplus', 'ReLU', 'ReLU6', '4', 'Exp']
positive_funs_chol = positive_funs + ['None']
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
        for pos in positive_funs_chol:
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
        try:
            _ = Eigen(n_zero_eigvals=100)
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

    def test_example_one(self):
        hidden_size = 100
        n_features = 2
        out_shape = 6
        in_shape = in_shape_from(out_shape)

        model = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.Linear(hidden_size, in_shape),
            Cholesky(output_shape=out_shape)
        )
        x = torch.rand((10, n_features))
        model(x)

    def test_example_two(self):
        hidden_size = 100
        n_features = 2
        out_shape = 6
        in_shape = in_shape_from(out_shape)

        model = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.Linear(hidden_size, in_shape),
            Eigen(output_shape=out_shape)
        )
        x = torch.rand((10, n_features))
        model(x)

    def test_different_output(self):
        hidden_size = 100
        n_features = 2
        out_shape = 21
        in_shape = in_shape_from(out_shape)

        model = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.Linear(hidden_size, in_shape),
            Eigen(output_shape=out_shape)
        )
        x = torch.rand((10, n_features))
        model(x)

    def test_training_eig_n_zero(self):
        in_shape = 6
        n_epochs = 5
        n_data = 10
        hidden_size = 10
        out_shape = 3
        n_features = 2
        X = torch.rand(n_data, n_features)
        Y = torch.zeros(n_data, out_shape, out_shape)
        Y[:, 0, 0] = 2.0*X[:, 0]
        Y[:, 1, 0] = -2.0*X[:, 1]
        Y[:, 1, 1] = 2.0*X[:, 0]*X[:, 1]
        Y[:, 2, 0] = -2.0*X[:, 0]
        Y[:, 2, 1] = 2.0*X[:, 1]
        Y[:, 2, 2] = 3.0 + X[:, 0]*X[:, 1]
        Y = Y @ Y.transpose(-2, -1)
        # Y is SPD
        X = X.double()
        Y = Y.double()

        # define the model using the Eigen SPD layer
        model = nn.Sequential(
                nn.Linear(n_features, hidden_size),
                nn.Linear(hidden_size, in_shape),
                Eigen(output_shape=out_shape,
                      symmetry='anisotropic',
                      positive='ReLU',
                      n_zero_eigvals=1,
                      min_value=0.0)
                )
        model = model.double()
        loss_fn = nn.MSELoss()
        loss0 = loss_fn(model(X), Y)
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0)

        for epoch in range(n_epochs):
            def closure():
                optimizer.zero_grad()
                output = model(X)
                loss = loss_fn(output, Y)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            print(f'Epoch: {epoch:4d} Loss: {loss.item():.5f}')
        self.assertTrue(loss.item() < loss0.item())


if __name__ == '__main__':
    unittest.main()
