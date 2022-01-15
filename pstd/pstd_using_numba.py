"""
This module contains a Numba-accelerated implementation of the k-space PSTD method.
"""

import numba
from . import pstd

kappa = numba.jit(pstd.kappa)
# abs_exp = numba.jit(pstd.abs_exp)
pressure_abs_exp = numba.jit(pstd.pressure_abs_exp)
velocity_abs_exp = numba.jit(pstd.velocity_abs_exp)
pressure_with_pml = numba.jit(pstd.pressure_with_pml)
velocity_with_pml = numba.jit(pstd.pressure_with_pml)
to_pressure_gradient = numba.jit(pstd.to_pressure_gradient)
to_velocity_gradient = numba.jit(pstd.to_velocity_gradient)
update = numba.jit(pstd.update)


class PSTD(pstd.PSTD):

    _update = staticmethod(update)

    # _record = numba.jit(pstd.PSTD._record)

    # kappa = staticmethod(numba.jit(PSTD.kappa))
    # abs_exp = staticmethod(numba.jit(PSTD.abs_exp))
    # pressure_with_pml = staticmethod(numba.jit(PSTD.pressure_with_pml))
    # velocity_with_pml = staticmethod(numba.jit(PSTD.velocity_with_pml))
    # pressure_gradient = staticmethod(numba.jit(PSTD.to_pressure_gradient))
    # velocity_gradient = staticmethod(numba.jit(PSTD.to_velocity_gradient))
    # update = classmethod(numba.jit(PSTD.update))

    # pre_run = numba.jit(PSTD.pre_run)

    # run = numba.jit(Model.run)
