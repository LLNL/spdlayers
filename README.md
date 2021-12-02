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

If you find this work useful, please cite our upcoming paper.

## License

see [LICENSE](https://github.com/LLNL/spdlayers/blob/main/LICENSE) and [NOTICE](https://github.com/LLNL/spdlayers/blob/main/NOTICE)

SPDX-License-Identifier: MIT

LLNL-CODE-829369
