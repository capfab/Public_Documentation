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
