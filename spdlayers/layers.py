# Copyright 2021, Lawrence Livermore National Security, LLC and spdlayer
# contributors
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn


def _positive_function(positive):
    """
    Returns the torch function belonging to a positive string
    """
    if positive == 'Abs':
        return torch.abs
    elif positive == 'Square':
        return torch.square
    elif positive == 'Softplus':
        return torch.nn.Softplus()
    elif positive == 'ReLU':
        return torch.nn.ReLU()
    elif positive == 'ReLU6':
        return torch.nn.ReLU6()
    elif positive == '4':
        return lambda x: torch.pow(x, 4)
    elif positive == 'Exp':
        return torch.exp
    else:
        error = f"Positve transformation {positive} not supported!"
        raise ValueError(error)


def _anisotropic_indices(output_shape):
    """
    Returns anisotropic indices to transform vector to matrix
    """
    inds_a, inds_b = torch.tril_indices(output_shape, output_shape)
    return inds_a, inds_b


def _orthotropic_indices():
    """
    Returns orthotropic indices to transform vector to matrix
    """
    inds_a = torch.tensor([0, 1, 1, 2, 2, 2, 3, 4, 5])
    inds_b = torch.tensor([0, 0, 1, 0, 1, 2, 3, 4, 5])
    return inds_a, inds_b


class Cholesky(nn.Module):
    """
    Symmetric Positive Definite (SPD) Layer via Cholesky Factorization
    """

    def __init__(self, output_shape=6, symmetry='anisotropic',
                 positive='Square', min_value=1e-8):
        """
        Initialize Cholesky SPD layer

        This layer takes a vector of inputs and transforms it to a Symmetric
        Positive Definite (SPD) matrix, for each candidate within a batch.

        Args:
            output_shape (int): The dimension of square tensor to produce,
                default output_shape=6 results in a 6x6 tensor
            symmetry (str): 'anisotropic' or 'orthotropic'. Anisotropic can be
                used to predict for any shape tensor, while 'orthotropic' is a
                special case of symmetry for a 6x6 tensor.
            positive (str): The function to perform the positive
                transformation of the diagonal of the lower triangle tensor.
                Choices are 'Abs', 'Square' (default), 'Softplus', 'ReLU',
                'ReLU6', '4', and 'Exp'.
            min_value (float): The minimum allowable value for a diagonal
                component. Default is 1e-8.
        """
        super(Cholesky, self).__init__()
        if symmetry == 'anisotropic':
            self.inds_a, self.inds_b = _anisotropic_indices(output_shape)
        elif symmetry == 'orthotropic':
            self.inds_a, self.inds_b = _orthotropic_indices()
            if output_shape != 6:
                e = f"symmetry={symmetry} can only be used with output_shape=6"
                raise ValueError(e)
        else:
            raise ValueError(f"Symmetry {symmetry} not supported!")
        self.is_diag = self.inds_a == self.inds_b
        self.output_shape = output_shape

        self.positive = positive
        self.positive_fun = _positive_function(positive)
        self.min_value = torch.tensor(min_value)
        self.register_buffer('_min_value', self.min_value)

    def forward(self, x):
        """
        Generate SPD tensors from x

        Args:
            x (Tensor): Tensor to generate predictions for. Must have
                2d shape of form (:, input_shape). If symmetry='anisotropic',
                the expected
                `input_shape = sum([i for i in range(output_shape + 1)])`. If
                symmetry='orthotropic', then the expected `input_shape=9`.

        Returns:
            (Tensor): The predictions of the neural network. Will return
                shape (:, output_shape, output_shape)
        """
        # enforce positive values for the diagonal
        x = torch.where(self.is_diag, self.positive_fun(x) + self.min_value, x)
        # init a Zero lower triangle tensor
        L = torch.zeros((x.shape[0], self.output_shape, self.output_shape),
                        dtype=x.dtype)
        # populate the lower triangle tensor
        L[:, self.inds_a, self.inds_b] = x
        LT = L.transpose(1, 2)  # lower triangle transpose
        out = torch.matmul(L, LT)  # return the SPD tensor
        return out


class Eigen(nn.Module):
    """
    Symmetric Positive Definite Layer via Eigendecomposition
    """

    def __init__(self, output_shape=6, symmetry='anisotropic',
                 positive='Square', min_value=1e-8):
        """
        Initialize Eigendecomposition SPD layer

        This layer takes a vector of inputs and transforms it to a Symmetric
        Positive Definite (SPD) matrix, for each candidate within a batch.

        Args:
            output_shape (int): The dimension of square tensor to produce,
                default output_shape=6 results in a 6x6 tensor
            symmetry (str): 'anisotropic' or 'orthotropic'. Anisotropic can be
                used to predict for any shape tensor, while 'orthotropic' is a
                special case of symmetry for a 6x6 tensor.
            positive (str): The function to perform the positive
                transformation of the diagonal of the lower triangle tensor.
                Choices are 'Abs', 'Square' (default), 'Softplus', 'ReLU',
                'ReLU6', '4', and 'Exp'.
            min_value (float): The minimum allowable value for a diagonal
                component. Default is 1e-8.
        """
        super(Eigen, self).__init__()
        if symmetry == 'anisotropic':
            self.inds_a, self.inds_b = _anisotropic_indices(output_shape)
        elif symmetry == 'orthotropic':
            self.inds_a, self.inds_b = _orthotropic_indices()
            if output_shape != 6:
                e = f"symmetry={symmetry} can only be used with output_shape=6"
                raise ValueError(e)
        else:
            raise ValueError(f"Symmetry {symmetry} not supported!")

        self.output_shape = output_shape
        self.positive = positive
        self.positive_fun = _positive_function(positive)
        self.min_value = torch.tensor(min_value)
        self.register_buffer('_min_value', self.min_value)

    def forward(self, x):
        """
        Generate SPD tensors from x

        Args:
            x (Tensor): Tensor to generate predictions for. Must have
                2d shape of form (:, input_shape). If symmetry='anisotropic',
                the expected
                `input_shape = sum([i for i in range(output_shape + 1)])`. If
                symmetry='orthotropic', then the expected `input_shape=9`.

        Returns:
            (Tensor): The predictions of the neural network. Will return
                shape (:, output_shape, output_shape)
        """
        x = torch.nan_to_num(x)  # we can't run torch.linalg.eig with NaNs!
        # init a placeholder tensor
        out = torch.zeros((x.shape[0], self.output_shape, self.output_shape),
                          dtype=x.dtype)
        out[:, self.inds_a, self.inds_b] = x
        out[:, self.inds_b, self.inds_a] = x

        # U, D, UT = torch.linalg.svd(out)  # SVD DOES NOT WORK! reason unknown
        D, U = torch.linalg.eig(out)
        U = torch.real(U)
        D = torch.real(D)
        UT = U.inverse()  # don't tranpose, need inverse!
        D = self.positive_fun(D) + self.min_value
        out = U @ torch.diag_embed(D) @ UT
        return out
