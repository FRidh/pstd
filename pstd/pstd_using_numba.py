"""
This module contains a Numba-accelerated implementation of the k-space PSTD method.
"""

import numba
from . import pstd

kappa = numba.autojit(pstd.kappa)
#abs_exp = numba.autojit(pstd.abs_exp)
pressure_abs_exp = numba.autojit(pstd.pressure_abs_exp)
velocity_abs_exp = numba.autojit(pstd.velocity_abs_exp)
pressure_with_pml = numba.autojit(pstd.pressure_with_pml)
velocity_with_pml = numba.autojit(pstd.pressure_with_pml)
to_pressure_gradient = numba.autojit(pstd.to_pressure_gradient)
to_velocity_gradient = numba.autojit(pstd.to_velocity_gradient)
update = numba.autojit(pstd.update)

class PSTD(pstd.PSTD):
    
    _update = staticmethod(update)
    
    #_record = numba.autojit(pstd.PSTD._record)
    
    #kappa = staticmethod(numba.autojit(PSTD.kappa))
    #abs_exp = staticmethod(numba.autojit(PSTD.abs_exp))
    #pressure_with_pml = staticmethod(numba.autojit(PSTD.pressure_with_pml))
    #velocity_with_pml = staticmethod(numba.autojit(PSTD.velocity_with_pml))
    #pressure_gradient = staticmethod(numba.autojit(PSTD.to_pressure_gradient))
    #velocity_gradient = staticmethod(numba.autojit(PSTD.to_velocity_gradient))
    #update = classmethod(numba.autojit(PSTD.update))
    
    #pre_run = numba.autojit(PSTD.pre_run)
    
    #run = numba.autojit(Model.run)

      
