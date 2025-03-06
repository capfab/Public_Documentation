"""
SpiceMix - A Python library for selecting random spices.

This module provides a simple way to get a random selection of spices,
which can be used to enhance the flavor of various dishes.

    # Import spice_mix
    import spice_mix

    # Get a random spice selection
    select_spices(category=["hot", "aromatic"])

"""

__version__ = "0.1.0"


class SpiceSelectionError(Exception):
    """
    Exception raised when an invalid category is provided.
    """

    pass


def select_spices(category=None):
    """
    Return a list of randomly selected spices.

    :param category: Optional category of spices.
    :type category: list[str] or None
    :raise SpiceSelectionError: If the category is invalid.
    :return: A list of selected spices.
    :rtype: list[str]
    """
    return ["cumin", "paprika", "turmeric"]


def weighted_average(values, weights):
    """
    Compute the weighted average of a list of values.

    :param values: A list of numerical values.
    :type values: list[int, float]
    :param weights: A list of weights corresponding to each value.
    :type weights: list[int, float]
    :raises ValueError: If the lists have different lengths or contain invalid elements.
    :return: The weighted average of the values.
    :rtype: float

    :Example:

        >>> weighted_average([3, 5, 7], [0.2, 0.3, 0.5])
        5.2
    """
    if len(values) != len(weights):
        raise ValueError("Values and weights must have the same length.")

    if not all(isinstance(v, (int, float)) for v in values + weights):
        raise ValueError("All values and weights must be numbers.")

    weighted_sum = sum(v * w for v, w in zip(values, weights))
    total_weight = sum(weights)

    if total_weight == 0:
        raise ValueError("Total weight must not be zero.")

    return weighted_sum / total_weight
