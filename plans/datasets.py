"""
Datasets - Python library for cooks and food lovers.
This is a Python docstring, we can use reStructuredText syntax here!
.. code-block:: python
    # Import Datasets
    import datasets
    # Call its only function
    datasets.my_function(kind=["cheeses"])
"""

__version__ = "0.1.0"


class MyClass:
    """A new Object"""
    
    def __init__(self, s_name="MyName"):
        """
        Initiation of the MyClass object.
        
        :param s_name: Name of object.
        :type s_name: str
        """
        self.name = s_name
        print(self.name)
    
    def do_stuff(self, s_str1, n_value):
        """
        A demo method.
        
        :param s_str1: string to print.
        :type s_str1: str
        :param n_value: value to print.
        :type n_value: float
        :return: a concatenated string
        :rtype: str
        
        """
        s_aux = s_str1 + str(n_value)
        return s_aux


def my_function(kind=None):
    """
    Return a list of random ingredients as strings.
    :param kind: Optional "kind" of ingredients.
    :type kind: list[str] or None
    :raise lumache.InvalidKindError: If the kind is invalid.
    :return: The ingredients list.
    :rtype: list[str]
    """
    return ["shells", "gorgonzola", "parsley"]
