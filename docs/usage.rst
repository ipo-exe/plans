Usage
=====

.. _installation:

Installation
------------

To use Lumache, first install it using pip:

.. code-block:: console

   (.venv) $ pip install lumache


Python code
.. code-block:: python

   print("hello world")

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

Referencing things in other pages
---------------------------------

Lorem ipsum dolor sit amet :ref:`some disadvantages <my target>`:, consectetur adipiscing elit. Integer eu egestas ipsum. Curabitur aliquam, nulla eget ornare commodo, nisl lectus auctor felis, quis facilisis libero justo ac nisl. As illustrated in :numref:`myfig` and more.
