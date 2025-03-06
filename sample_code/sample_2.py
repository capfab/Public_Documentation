def exponentiate(base: float, exponent: float) -> float:
    """
    Compute the exponentiation of a number.

    This function raises the given base to the power of the specified exponent.

    Args:
        base (float): The base number.

        exponent (float): The exponent to which the base is raised.

    Returns:
        float: The result of base raised to the power of exponent.

    Example:

        >>> exponentiate(2, 3)
        8.0
        >>> exponentiate(5, -1)
        0.2
    """
    return base**exponent
