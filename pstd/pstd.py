"""
This module contains an implementation of the k-space PSTD method.
"""
import numpy as np

import logging
logger = logging.getLogger(__name__)    # Use module name as logger name


try:
    from pyfftw.interfaces.numpy_fft import fft2, ifft2  # Performs much better than numpy's fftpack
except ImportError:                                    # Use monkey-patching np.fft perhaps instead?
    from numpy.fft import fft2, ifft2


#class AttributeDict(collections.OrderedDict):
    #"""
    #Dictionary that allows accessing the items through attribute-access (e.g. a.x).
    #"""
    #def __init__(self,**kw):
        #dict.__init__(self,kw)
        #self.__dict__ = self


from model import Model    
    

def kappa(wavenumber, timestep, c):
    """
    k-space operator.
    
    :param wavenumber: Wavenumber :math:`k`
    :param timestep: Timestep :math:`\\Delta t`
    
    .. math:: \\kappa = \\mathrm{sinc}{\\left( c_0 \\Delta t k / 2 \\right)}
    
    """
    return np.sinc(c * timestep * wavenumber / 2.0)

def to_pressure_gradient(pressure_fft, wavenumber, kappa, spacing):
    """
    Pressure gradient.
    
    :param wavenumber: Wavenumber :math:`k`
    :param kappa: k-space operator :math:`\\kappa`
    :param spacing: Spacing :math:`\\Delta \\xi`
    :param pressure: Pressure at timestep :math:`p^n`
    
    .. math:: \\frac{\\partial }{\\partial \\xi} p^n = \\mathcal{F}^{-1} \\{ i k_{\\xi} \\kappa e^{i k_{\\xi} \\Delta \\xi / 2 } \\mathcal{F} \{ p^n \} \} 
    
    K-space documentation Equation 2.17a as well as 
    """
    #return ifft(+1j * wavenumber * kappa * np.exp(+1j*wavenumber*spacing/2.0) * fft(pressure, axis=axis), axis=axis)
    return (+1j * wavenumber * kappa * np.exp(+1j*wavenumber*spacing/2.0) * pressure_fft)#fft2(pressure))


def to_velocity_gradient(velocity_fft, wavenumber, kappa, spacing):
    """
    Velocity gradient.
    
    :param wavenumber: Wavenumber :math:`k`
    :param kappa: k-space operator :math:`\\kappa`
    :param spacing: Spacing :math:`\\Delta \\xi`
    :param velocity: Pressure at timestep :math:`u_{\\xi}^{n+\\frac{1}{2}}`
    
    .. math:: \\frac{\\partial }{\\partial \\xi} u_{\\xi}^{n+\\frac{1}{2}} = \\mathcal{F}^{-1} \\{i k_{\\xi} \\kappa e^{-i k_{\\xi} \\Delta \\xi / 2 } \\mathcal{F} \\{ u_{\\xi}^{n+\\frac{1}{2}} \} \} 
    
    Equation 2.17c.
    """
    #return ifft(+1j * wavenumber * kappa * np.exp(-1j*wavenumber*spacing/2.0) * fft(velocity, axis=axis), axis=axis)
    return (+1j * wavenumber * kappa * np.exp(-1j*wavenumber*spacing/2.0) * velocity_fft)


def abs_exp(alpha, timestep):
    """
    Absorption coefficient exponent.
    
    :param alpha: :math:`\\alpha_{\\xi}`
    :param timestep: Timestep :math:`\\Delta t`
    
    This value is calculated according to
    
    .. math:: e^{-\\alpha_{\\xi} \\Delta t / 2}
    
    """
    return np.exp(alpha * -timestep / 2.0)

def pressure_abs_exp(alpha, timestep):
    """
    Absorption coefficient exponent.
    
    :param alpha: :math:`\\alpha_{\\xi}`
    :param timestep: Timestep :math:`\\Delta t`
    
    This value is calculated according to
    
    .. math:: e^{-\\alpha_{\\xi} \\Delta t / 2}
    
    """
    return np.exp(alpha * -timestep / 2.0)

def velocity_abs_exp(alpha, timestep, spacing, wavenumber):
    """
    Absorption coefficient exponent.
    
    :param alpha: :math:`\\alpha_{\\xi}`
    :param timestep: Timestep :math:`\\Delta t`
    
    This value is calculated according to
    
    .. math:: e^{-\\alpha_{\\xi} \\Delta t / 2}
    
    """
    return ifft2(np.exp(+1j*wavenumber*spacing/2.0) * fft2(np.exp(alpha * -timestep / 2.0)) )

def velocity_with_pml(previous_velocity, pressure_gradient, timestep, density, abs_exp, source):
    """
    Velocity.
    
    :param previous_velocity: Velocity at previous timestep.
    :param pressure_gradient:  Pressure gradient at previous timestep.
    :param timestep: Timestep :math:`\\Delta t`
    :param density: Density :math:`\\rho_0`
    :param abs_exp: Absorption exponent :math:`e^{-\\alpha_{\\xi} \\Delta t / 2}`
    :param source: Source term :math:`S^n_{F_{\\xi}}`
    
    
    .. math:: u_{\\xi}^{n+\\frac{1}{2}} = e^{-\\alpha_{\\xi} \\Delta t / 2} \\left( e^{-\\alpha_{\\xi} \\Delta t / 2} u_{\\xi}^{n-\\frac{1}{2}} - \\frac{\\Delta t}{\\rho_0} \\frac{\\partial}{\\partial \\xi} p^n + \\Delta t S^n_{F_{\\xi}} \\right)
    
    Equation 2.27.
    """
    return abs_exp * (abs_exp * previous_velocity - timestep / density * pressure_gradient  + timestep * source) 


def pressure_with_pml(previous_pressure, velocity_gradient, timestep, density, soundspeed, abs_exp, source):
    """
    Pressure.
    
    :param previous_pressure: Pressure at previous timestep.
    :param velocity_gradient: Velocity gradient at previous timestep.
    :param timestep: Timestep :math:`\\Delta t`
    :param density: Density :math:`\\rho_0`
    :param soundspeed: Speed of sound :math:`c`.
    :param abs_exp: Absorption exponent :math:`e^{-\\alpha_{\\xi} \\Delta t / 2}`
    :param source: Source term :math:`S^n_{M_{\\xi}}`.
    
    .. math:: p_{\\xi}^{n+1} = e^{-\\alpha_{\\xi} \\Delta t / 2} \\left( e^{-\\alpha_{\\xi} \\Delta t / 2} u_{\\xi}^{n} - \\Delta t \\rho_0 c^2 \\frac{\\partial}{\\partial \\xi} v^n + \\Delta t S^n_{M_{\\xi}} \\right)
    
    """
    return abs_exp * (abs_exp * previous_pressure - timestep * (density * soundspeed**2.0)  * velocity_gradient + timestep * source)

#@classmethod
def update(d):
    """
    Calculation steps to be taken every step. 
    
    :param d: Dictionary containing simulation data.
    
    .. note:: This method should only contain calculation steps.
    
    """
    #d_p_d_x = cls.pressure_gradient(p_x + p_y, k_x, kappa, spacing)
    #d_p_d_y = cls.pressure_gradient(p_y + p_y, k_y, kappa, spacing)
    
    step = d['step']
    
    print "Step: {}".format(step)
    
    pressure_fft = fft2(d['field']['p'])    # Apply atmospheric absorption here?
    
    #pressure_fft *= data['absorption']
    
    d['field']['v_x'] = velocity_with_pml(d['field']['v_x'], 
                                                ifft2(to_pressure_gradient(pressure_fft, 
                                                                    d['k_x'], d['kappa'], d['spacing'])), 
                                                                    d['timestep'], d['density'], d['abs_exp']['v']['x'], d['source']['v']['x'][step])
                                                                    #d['timestep'], d['density'], d['abs_exp']['x'], d['source']['v']['x'][step])
    d['field']['v_y'] = velocity_with_pml(d['field']['v_y'], 
                                                ifft2(to_pressure_gradient(pressure_fft,
                                                                    d['k_y'], d['kappa'], d['spacing'])), 
                                                                    d['timestep'], d['density'], d['abs_exp']['v']['y'], d['source']['v']['y'][step])
                                                                    #d['timestep'], d['density'], d['abs_exp']['y'], d['source']['v']['y'][step])
    
    #print d['field']['v_x']
        
    #d_v_d_x = cls.velocity_gradient(v_x, k_x, kappa, spacing)
    #d_v_d_y = cls.velocity_gradient(v_y, k_y, kappa, spacing)
    
    d['temp']['p_x'] = pressure_with_pml(d['temp']['p_x'], 
                                                ifft2(to_velocity_gradient(fft2(d['field']['v_x']), d['k_x'], 
                                                                    d['kappa'], d['spacing'])), d['timestep'], 
                                                                    d['density'], d['soundspeed'], d['abs_exp']['p']['x'], d['source']['p'][step])
                                                                    #d['density'], d['soundspeed'], d['abs_exp']['x'], d['source']['p'][step])
    d['temp']['p_y'] = pressure_with_pml(d['temp']['p_y'], 
                                                ifft2(to_velocity_gradient(fft2(d['field']['v_y']), d['k_y'], 
                                                                    d['kappa'], d['spacing'])), d['timestep'], 
                                                                    d['density'], d['soundspeed'], d['abs_exp']['p']['y'], d['source']['p'][step])
                                                                    #d['density'], d['soundspeed'], d['abs_exp']['y'], d['source']['p'][step])
    
    
    d['field']['p'] = d['temp']['p_x'] + d['temp']['p_y']
    
    #print "Source p: {}".format(d['source']['p'][step])
    
    #print "Velocity y: {}".format(d['field']['v_y'])
    
    #print "Pressure total: {}".format(d['field']['p'])    


class PSTD(Model):
    """
    K-space Pseudo Spectral Time-Domain model.
    """
    
    FIELD_ARRAYS = ['p', 'v_x', 'v_y']
    
    _update = staticmethod(update)
    
    @staticmethod
    def stability_criterion(CFL, c_0, c_ref):
        """
        K-space PSTD stability criterium as function of CFL.
        
        :param CFL: CFL
        :param c_0: Speed of sound field values :math:`c_0`
        :param c_ref: Reference speed of sound :math:`c_{ref}`
        
        .. math:: CFL <= \\frac{2}{\\pi} \\frac{c_0}{c_{ref}} \\sin^{-1}{\\left(\\frac{c_{ref}}{c_{0}} \\right)}
        
        """
        return CFL <= 2.0 / np.pi * (c_0 / c_ref) * np.arcsin(c_ref/c_0)

    def _pre_run(self, data):
        
        super(PSTD, self)._pre_run(data)
        
        data['k_x'], data['k_y'] = np.meshgrid(self.axes.x.wavenumbers, self.axes.y.wavenumbers, indexing='ij')
        data['k_x'] = data['k_x'].astype(self.dtype('float'))
        data['k_y'] = data['k_y'].astype(self.dtype('float'))
        
        data['k'] = np.sqrt(data['k_x']**2.0 + data['k_y']**2.0)
        
        data['kappa'] = kappa(data['k'], data['timestep'], np.mean(self.medium.soundspeed))         # Independent of time when considering a frozen field.
        data['density'] = self.medium.density * np.ones(self.grid.shape, dtype=self.dtype('float'))
        data['soundspeed'] = (self.medium.soundspeed_for_calculation * np.ones(self.grid.shape) ).astype(self.dtype('float'))
        
        data['abs_exp'] = {'p': dict(), 'v': dict()}
        data['abs_exp']['p']['x'] = pressure_abs_exp(data['pml']['x'], data['timestep'])     # Absorption exponent for x-direction
        data['abs_exp']['p']['y'] = pressure_abs_exp(data['pml']['y'], data['timestep'])     # Absorption exponent for y-direction
        data['abs_exp']['v']['x'] = velocity_abs_exp(data['pml']['x'], data['timestep'], data['spacing'], data['k_x'])     # Absorption exponent for x-direction
        data['abs_exp']['v']['y'] = velocity_abs_exp(data['pml']['y'], data['timestep'], data['spacing'], data['k_y'])     # Absorption exponent for y-direction
        
        
        data['size'] = self.grid.size
        data['shape'] = self.grid.shape
        
        data['temp'] = dict()
        data['temp']['p_x'] = np.zeros_like(data['field']['p'])
        data['temp']['p_y'] = np.zeros_like(data['field']['p'])
        
        
        #from acoustics.atmosphere import Atmosphere
        #atm = Atmosphere()
        #L_alpha = atm.attenuation_coefficient(self._frequencies)
        #data['absorption'] = 10.0**(-L_alpha/20.0)
        
        
    