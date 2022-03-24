# Copyright 2021, Lawrence Livermore National Security, LLC and spdlayer
# contributors
# SPDX-License-Identifier: MIT

def in_shape_from(output_shape):
    """
    Returns input_shape required for a output_shape x output_shape SPD tensor

    Args:
        output_shape (int): The dimension of square tensor to produce.

    Returns:
        (int): The input shape associated from a
            `[output_shape, output_shape` SPD tensor.

    Notes:
        This is the sum of the first n natural numbers
        https://cseweb.ucsd.edu/groups/tatami/handdemos/sum/
    """
    input_shape = output_shape * (output_shape + 1) // 2
    return input_shape
