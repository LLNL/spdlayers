# spdlayers

Symmetric Positive Definite (SPD) enforcement layers for PyTorch.

Regardless of the input, the output of these layers will always be a SPD tensor!

## Installation

Install with pip

```
python -m pip install spdlayers
```

## About

The `Cholesky` layer uses a cholesky factorization to enforce SPD, and the `Eigen` layer uses an eigendecomposition to enforce SPD.

Both layers take in some tensor of shape `[batch_size, input_shape]` and output a SPD tensor of shape `[batch_size, output_shape, output_shape]`. The relationship between input and output is defined by the following.

```python
input_shape = sum([i for i in range(output_shape + 1)])
```

The layers have no learnable parameters, and merely serve to transform a vector space to a SPD matrix space.

The initialization options for each layer are:
```
Args:
    output_shape (int): The dimension of square tensor to produce,
        default output_shape=6 results in a 6x6 tensor
    symmetry (str): 'anisotropic' or 'orthotropic'. Anisotropic can be
        used to predict for any shape tensor, while 'orthotropic' is a
        special case of symmetry for a 6x6 tensor.
    positive (str): The function to perform the positive
        transformation of the diagonal of the lower triangle tensor.
        Choices are 'Abs' (default), 'Square', 'Softplus', 'ReLU',
        'ReLU6', '4', and 'Exp'.
    min_value (float): The minimum allowable value for a diagonal
        component. Default is 1e-8.
```

## Examples

This is the simplest neural network using 1 hidden layer of size 100. There are 2 input features to the model (`n_features = 2`), and model outputs a `6 x 6` spd tensor.

Using the Cholesky factorization as the SPD layer:
```python
import torch.nn as nn
import spdlayers

hidden_size = 100
n_features = 2
out_shape = 6
in_shape = spdlayers.in_shape_from(out_shape)

model = nn.Sequential(
          nn.Linear(n_features, hidden_size),
          nn.Linear(hidden_size, in_shape),
          spdlayers.Cholesky(output_shape=out_shape)
        )
```

Or with the eigendecomposition as the SPD layer:
```python
import torch.nn as nn
import spdlayers

hidden_size = 100
n_features = 2
out_shape = 6
in_shape = spdlayers.in_shape_from(out_shape)

model = nn.Sequential(
          nn.Linear(n_features, hidden_size),
          nn.Linear(hidden_size, in_shape),
          spdlayers.Eigen(output_shape=out_shape)
        )
```

[examples/train_sequential_model.py](https://github.com/LLNL/spdlayers/blob/main/examples/train_sequential_model.py) trains this model on the orthotropic stiffness trensor from the 2D Isotruss.

## API

The API has the following import structure.

```
spdlayers
    ├── Cholesky
    ├── Eigen
    ├── in_shape_from
    ├── layers
    │   ├── Cholesky
    │   ├── Eigen
    ├── tools.py
    │   ├── in_shape_from
```

## Documentation

You can use pdoc to build API documentation, or view the [online documentation](https://software.llnl.gov/spdlayers).

```
pdoc3 --html spdlayers
```

## Requirements

For basic usage:

```
python>=3.6
torch>=1.9.0
```

Additional dependencies for testing:

```
pytest
pytest-cov
numpy
```

## Changelog

Changes are documented in [CHANGELOG.md](https://github.com/LLNL/spdlayers/blob/main/CHANGELOG.md)

## Citation

### Cholesky

If you use the Cholesky method, you should cite the following paper.

```bib
@article{XU2021110072,
title = {Learning constitutive relations using symmetric positive definite neural networks},
journal = {Journal of Computational Physics},
volume = {428},
pages = {110072},
year = {2021},
issn = {0021-9991},
doi = {https://doi.org/10.1016/j.jcp.2020.110072},
url = {https://www.sciencedirect.com/science/article/pii/S0021999120308469},
author = {Kailai Xu and Daniel Z. Huang and Eric Darve},
keywords = {Neural networks, Plasticity, Hyperelasticity, Finite element method, Multiscale homogenization}
}
```

What this paper proposed is that you do a Cholesky decomposition on each data point in your dataset, and then train a NN to learn the lower triangular form. This is a great paper because the method is so simple that it is very easy for every researcher to use!

What we proposed you do instead with this method is not to perform a Cholesky decomposition on your data, but rather include the `LL^T` operation as a transformation within your NN. We believe this subtle difference provides for the following advantages:
- no transformation of your dataset is needed
- evaluate the training performance on the real data
- eases production use of the model (i.e. the black-box mapping of `x`→`C`)

### Eigendecomposition

If you use the Eigendecomposition method, you should cite the following paper.

```bib
@article{https://doi.org/10.1002/nme.2681,
author = {Amsallem, David and Cortial, Julien and Carlberg, Kevin and Farhat, Charbel},
title = {A method for interpolating on manifolds structural dynamics reduced-order models},
journal = {International Journal for Numerical Methods in Engineering},
volume = {80},
number = {9},
pages = {1241-1258},
keywords = {reduced-order modeling, matrix manifolds, real-time prediction, surrogate modeling, linear structural dynamics},
doi = {https://doi.org/10.1002/nme.2681},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.2681},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/nme.2681}
}
```
What this paper proposed is that you perform an Eigendecomposition on each data point in your dataset. You could then fit a linear model in a tangent space that will always be SPD (and it is pure mathematical beauty).

We proposed you should abstract this method into a NN layer. This subtle change allows for the following advantages:
- no transformation of your dataset is needed
- evaluate the training performance in the real space
- easily fit any non-linear regression model
- eases production use of the model (i.e. the black-box mapping of `x`→`C`)

### spdlayers

If you find our abstractions and presentation useful, please cite our paper. Our paper demonstrated that the inclusion of these `spdlayers` resulted in more accurate models.

```bib
@article{jekel2022neural,
  title={Neural Network Layers for Prediction of Positive Definite Elastic Stiffness Tensors},
  author={Jekel, Charles F and Swartz, Kenneth E and White, Daniel A and Tortorelli, Daniel A and Watts, Seth E},
  journal={arXiv preprint arXiv:2203.13938},
  year={2022}
}
```

## License

see [LICENSE](https://github.com/LLNL/spdlayers/blob/main/LICENSE) and [NOTICE](https://github.com/LLNL/spdlayers/blob/main/NOTICE)

SPDX-License-Identifier: MIT

LLNL-CODE-829369
