
# Not used. ne.evaluate is now used in the main implementation!

from .pstd import PSTD

import numexpr as ne

def kappa(wavenumber, timestep, c):
    return ne.evaluate("sin(c * timestep * wavenumber / 2.0) / (c * timestep * wavenumber / 2.0)")

def to_pressure_gradient(pressure_fft, wavenumber, kappa, spacing):
    return ne.evaluate("+1j * wavenumber * kappa * exp(wavenumber*spacing/2.0 * +1j) * pressure_fft")

def to_velocity_gradient(velocity_fft, wavenumber, kappa, spacing):
    return ne.evaluate("+1j * wavenumber * kappa * exp(wavenumber*spacing/2.0 * -1j) * velocity_fft")

def abs_exp(alpha, timestep):
    return ne.evaluate("alpha * -timestep / 2.0")

def pressure_abs_exp(alpha, timestep):
    return ne.evaluate("exp(alpha * -timestep / 2.0)")

def velocity_abs_exp(alpha, timestep, spacing, wavenumber):
    return ifft2(ne.evaluate("exp(+1j*wavenumber*spacing/2.0)") * fft2(ne.evaluate("alpha * -timestep / 2.0)")) )

def velocity_with_pml(previous_velocity, pressure_gradient, timestep, density, abs_exp, source):
    return ne.evaluate("abs_exp * (abs_exp * previous_velocity - timestep / density * pressure_gradient  + timestep * source)")

def pressure_with_pml(previous_pressure, velocity_gradient, timestep, density, soundspeed, abs_exp, source):
    return ne.evaluate("abs_exp * (abs_exp * previous_pressure - timestep * (density * soundspeed**2.0)  * velocity_gradient + timestep * source)")


PSTD.kappa = kappa
PSTD.to_pressure_gradient = to_pressure_gradient
PSTD.to_velocity_gradient = to_velocity_gradient
PSTD.abs_exp = abs_exp
PSTD.pressure_abs_exp = pressure_abs_exp
PSTD.velocity_abs_exp = velocity_abs_exp
PSTD.velocity_with_pml = velocity_with_pml
PSTD.pressure_with_pml = pressure_with_pml

def update(d):
    """
    Calculation steps to be taken every step. 
    
    :param d: Dictionary containing simulation data.
    
    .. note:: This method should only contain calculation steps.
    
    """
    #d_p_d_x = cls.pressure_gradient(p_x + p_y, k_x, kappa, spacing)
    #d_p_d_y = cls.pressure_gradient(p_y + p_y, k_y, kappa, spacing)
    
    step = d['step']
    
    #print "Step: {}".format(step)
    
    pressure_fft = fft2(d['field']['p'])    # Apply atmospheric absorption here?
    
    #pressure_fft *= data['absorption']
    
    d['field']['v_x'] = velocity_with_pml(d['field']['v_x'], 
                                                ifft2(to_pressure_gradient(pressure_fft, 
                                                                    d['k_x'], d['kappa'], d['spacing'])), 
                                                                    d['timestep'], d['density'], d['abs_exp']['v']['x'], d['source']['v']['x'])
                                                                    #d['timestep'], d['density'], d['abs_exp']['x'], d['source']['v']['x'][step])
    d['field']['v_y'] = velocity_with_pml(d['field']['v_y'], 
                                                ifft2(to_pressure_gradient(pressure_fft,
                                                                    d['k_y'], d['kappa'], d['spacing'])), 
                                                                    d['timestep'], d['density'], d['abs_exp']['v']['y'], d['source']['v']['y'])
                                                                    #d['timestep'], d['density'], d['abs_exp']['y'], d['source']['v']['y'][step])
    
    #print d['field']['v_x']
        
    #d_v_d_x = cls.velocity_gradient(v_x, k_x, kappa, spacing)
    #d_v_d_y = cls.velocity_gradient(v_y, k_y, kappa, spacing)
    
    d['temp']['p_x'] = pressure_with_pml(d['temp']['p_x'], 
                                                ifft2(to_velocity_gradient(fft2(d['field']['v_x']), d['k_x'], 
                                                                    d['kappa'], d['spacing'])), d['timestep'], 
                                                                    d['density'], d['soundspeed'], d['abs_exp']['p']['x'], d['source']['p'])
                                                                    #d['density'], d['soundspeed'], d['abs_exp']['x'], d['source']['p'][step])
    d['temp']['p_y'] = pressure_with_pml(d['temp']['p_y'], 
                                                ifft2(to_velocity_gradient(fft2(d['field']['v_y']), d['k_y'], 
                                                                    d['kappa'], d['spacing'])), d['timestep'], 
                                                                    d['density'], d['soundspeed'], d['abs_exp']['p']['y'], d['source']['p'])
                                                                    #d['density'], d['soundspeed'], d['abs_exp']['y'], d['source']['p'][step])
    
    
    d['field']['p'] = d['temp']['p_x'] + d['temp']['p_y']
    
    return d


PSTD.update = staticmethod(update)
