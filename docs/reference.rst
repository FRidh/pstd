.. _reference:

Reference
#########

This section contains a description of all classes and functions.

.. inheritance-diagram:: pstd
.. automodule:: pstd
    :show-inheritance:
    :members:
    
.. toctree::
   :maxdepth: 2

=====
Model
=====

**********
Base class
**********

.. autoclass:: pstd.model.Model
    :show-inheritance:
    :members:
    :exclude-members: grid, pml, axes

.. autofunction:: pstd.model.DEFAULT_SETTINGS   
    
*************
Grid and axes
*************
.. autoclass:: pstd.model.Grid
    :show-inheritance:
    :members:
.. autoclass:: pstd.model.Axes
    :show-inheritance:
    :members:
.. autoclass:: pstd.model.Axes2D
    :show-inheritance:
    :members:
.. autoclass:: pstd.model.Axes3D
    :show-inheritance:
    :members:
.. autoclass:: pstd.model.Axis
    :show-inheritance:
    :members:

*********************
Sources and receivers
*********************

.. autoclass:: pstd.model.Source
    :show-inheritance:
    :members:
.. autoclass:: pstd.model.Receiver
    :show-inheritance:
    :members:
    
******
Medium
******
.. autoclass:: pstd.model.Medium
    :show-inheritance:
    :members:
    
********
Position
********
.. autoclass:: pstd.model.Position
    :show-inheritance:
    :members:
.. autoclass:: pstd.model.Position2D
    :show-inheritance:
    :members:
        
**********************
Pefectly Matched Layer
**********************
.. autoclass:: pstd.model.PML
    :show-inheritance:
    :members:
    
=========
Functions
=========

.. autofunction:: pstd.model.cfl

.. autofunction:: pstd.model.frequencies

.. autofunction:: pstd.model.wavenumbers

.. autofunction:: pstd.model.circular_receiver_array


==================
Models and kernels
==================


************
k-space PSTD
************

.. automodule:: pstd.pstd
    :show-inheritance:
    :members:

.. autoclass:: pstd.pstd.PSTD
    :show-inheritance:
    :members:

.. autoclass:: pstd.pstd_using_numba.PSTD
    :show-inheritance:
    :members:


    












    
