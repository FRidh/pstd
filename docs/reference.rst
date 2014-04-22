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

.. autoclass:: pstd.model.Model
    :show-inheritance:
    :members:
    :exclude-members: grid, pml, axes
    
*************
Grid and axes
*************
.. autoclass:: pstd.Grid
    :show-inheritance:
    :members:
.. autoclass:: pstd.Axes
    :show-inheritance:
    :members:
.. autoclass:: pstd.Axes2D
    :show-inheritance:
    :members:
.. autoclass:: pstd.Axes3D
    :show-inheritance:
    :members:
.. autoclass:: pstd.Axis
    :show-inheritance:
    :members:

*********************
Sources and receivers
*********************

.. autoclass:: pstd.Source
    :show-inheritance:
    :members:
.. autoclass:: pstd.Receiver
    :show-inheritance:
    :members:
    
******
Medium
******
.. autoclass:: pstd.Medium
    :show-inheritance:
    :members:
    
********
Position
********
.. autoclass:: pstd.Position
    :show-inheritance:
    :members:
.. autoclass:: pstd.Position2D
    :show-inheritance:
    :members:
.. autoclass:: pstd.Position3D
    :show-inheritance:
    :members:
        
**********************
Pefectly Matched Layer
**********************
.. autoclass:: pstd.PML
    :show-inheritance:
    :members:
    
=========
Functions
=========

.. autofunction:: pstd.CFL

.. autofunction:: pstd.frequencies

.. autofunction:: pstd.wavenumbers

.. autofunction:: pstd.initial_pressure_pulse

.. autofunction:: pstd.ir2fr

.. autofunction:: pstd.circular_receiver_array


***********
Conversions
***********

.. autofunction:: pstd.decibel_to_neper

.. autofunction:: pstd.neper_to_decibel

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

.. autoclass:: pstd.pstd_using_numba.PSTD_using_numba
    :show-inheritance:
    :members:


    












    