def in_shape_from(output_shape):
    """
    Returns input_shape required for a output_shape x output_shape SPD tensor

    Args:
        output_shape (int): The dimension of square tensor to produce.

    Returns:
        (int): The input shape associated from a
            `[output_shape, output_shape` SPD tensor.
    """
    input_shape = sum([i for i in range(output_shape + 1)])
    return input_shape