"""
This module contains a CUDA-accelerated implementation of the k-space PSTD method.
"""

import cmath
import math
import numpy as np

try:
    from numba import autojit
    from numbapro import vectorize, complex128 
    from numbapro import cuda
    from numbapro.cudalib.cufft import fft
    from numbapro.cudalib.cufft import ifft
except ImportError:
    raise ImportWarning("Cannot import numbapro. Cuda not available.") 

from pstd.pstd import PSTD

#vectorize(['complex128(complex128, float64, float64)'], target='cpu')(pstd.kappa)
#abs_exp = vectorize(['complex128(complex128, float64)'], target='cpu')(pstd.abs_exp)

@vectorize(['complex128(complex128, complex128, float64, float64, float64, complex128, float64)'], target='gpu')
def pressure_with_pml(previous_pressure, velocity_gradient, timestep, density, soundspeed, abs_exp, source):
    return abs_exp * (abs_exp * previous_pressure - timestep * (density * soundspeed**2.0)  * velocity_gradient + timestep * source)


@vectorize(['complex128(complex128, complex128, float64, float64, complex128, float64)'], target='gpu')
def velocity_with_pml(previous_velocity, pressure_gradient, timestep, density, abs_exp, source):
    return abs_exp * (abs_exp * previous_velocity - timestep / density * pressure_gradient  + timestep * source) 


#to_pressure_gradient = vectorize(['complex128(complex128, complex128, complex128, float64)'], target='gpu')(pstd.to_pressure_gradient)
#to_velocity_gradient = vectorize(['complex128(complex128, complex128, complex128, float64)'], target='gpu')(pstd.to_velocity_gradient)


#@vectorize(['complex128(complex128, complex128, complex128, float64)'], target='gpu')
#def to_pressure_gradient(pressure_fft, wavenumber, kappa, spacing):
    #return (complex128(1j) * wavenumber * kappa * cmath.exp(complex128(1j)*wavenumber*spacing/2.0) * pressure_fft)#fft2(pressure))


#@vectorize(['complex128(complex128, complex128, complex128, float64)'], target='gpu')
#def to_velocity_gradient(velocity_fft, wavenumber, kappa, spacing):
    #return (complex128(1j) * wavenumber * kappa * cmath.exp(complex128(-1j)*wavenumber*spacing/2.0) * velocity_fft)


@vectorize(["complex128(complex128, complex128)"], target='gpu')
def add(a, b):
    return a + b


@vectorize(["complex128(float64, float64)"], target='gpu')
def pressure_gradient_exponent(wavenumber, spacing):
    """
    Exponent in pressure gradient.
    """
    return +1j * wavenumber * spacing / 2.0

@vectorize(["complex128(float64, float64)"], target='gpu')
def velocity_gradient_exponent(wavenumber, spacing):
    """
    Exponent in velocity gradient.
    """
    return complex128(-1j) * wavenumber * spacing / complex128(2.0)

@vectorize(["complex128(complex128, float64, float64, complex128)"], target='gpu')
def to_gradient(transformed, wavenumber, kappa, exponent):
    """
    To Gradient.
    """
    return complex128(+1j) * wavenumber * kappa * exponent * transformed

@vectorize(["complex128(complex128)"], target='gpu')
def exp(r):
    """
    Complex exponent.
    """
    x = r.real
    y = r.imag
    return math.exp(x) * (math.cos(y) + complex128(1j) * math.sin(y) )

def dict_host_to_device(d):
    """
    Copy arrays from host to device.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_host_to_device(value)
        else:
            try:
                d[key] = cuda.to_device(value)
            except TypeError:
                pass
            except AttributeError:
                pass
    else:
        return d

def dict_device_to_host(d):
    """
    Copy arrays from device to host.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_device_to_host(value)
        else:
            try:
                d[key] = value.copy_to_host()
            except TypeError:
                pass
            except AttributeError:
                pass
    else:            
        return d
    

#def sync_steps(stream, p, v, p_fft, k, kappa, spacing, timestep, density, soundspeed, abs_exp_p, abs_exp_v, source_p, source_v):
    
    
    
    #v = velocity_with_pml(v, ifft2(to_pressure_gradient(p_fft, k, kappa, spacing)), timestep, density, abs_exp_v, source_v)
    #p = pressure_with_pml(p, ifft2(to_velocity_gradient(fft2(v), k, kappa, spacing)), timestep, density, soundspeed, abs_exp_p, source_p)

    #return p, v

 
class PSTD_using_cuda(PSTD):
    
    @property
    def precision(self):
        """
        Floating point precision.
        
        .. note:: Fixed to single precision.
        """
        return 'double'
    
    
    @staticmethod
    #@autojit
    def _update(d):
        
        stream1 = cuda.stream()
        stream2 = cuda.stream()
        stream3 = cuda.stream()
        stream4 = cuda.stream()
        
        step = d['step']
        
        #print "Step: {}".format(step)
        
        """Calculate the pressure gradient. Two steps are needed for this."""
        # Calculate FFT of pressure.
        fft(d['field']['p'], d['temp']['fft_p'], stream=stream1)    
        
        stream1.synchronize()
        #print "FFT pressure: {}".format(d['temp']['fft_p'].copy_to_host())
        
        #pressure_exponent_x = exp(pressure_gradient_exponent(d['k_x'], d['spacing'], stream=stream1), stream=stream1) # This is a constant!!
        #pressure_exponent_y = exp(pressure_gradient_exponent(d['k_y'], d['spacing'], stream=stream2), stream=stream2) # This is a constant!!
        
                
        #print(d['spacing'].shape)
        #print(d['k_x'].shape)
        
        ex = cuda.device_array(shape=d['field']['p'].shape)
        
        print(d['k_x'].shape)
        print(d['spacing'].shape)
        print(d['k_x'].dtype)
        print(d['spacing'].dtype)
        print(pressure_gradient_exponent(d['k_x'], d['spacing']))
        
        ex = pressure_gradient_exponent(d['k_x'], d['spacing'])#, stream=stream1)
        ey = pressure_gradient_exponent(d['k_y'], d['spacing'])#, stream=stream2)
        
        pressure_exponent_x = exp(ex, stream=stream1) # This is a constant!!
        pressure_exponent_y = exp(ey, stream=stream2) # This is a constant!!
        
        
        stream1.synchronize()
        stream2.synchronize()
        
        #print ( to_gradient(d['temp']['fft_p'], d['k_x'], d['kappa'], pressure_exponent_x) ).copy_to_host()
        
        """Calculate the velocity gradient."""
        ifft(to_gradient(d['temp']['fft_p'], d['k_x'], d['kappa'], pressure_exponent_x, stream=stream1), d['temp']['d_p_d_x'], stream=stream1)
        ifft(to_gradient(d['temp']['fft_p'], d['k_y'], d['kappa'], pressure_exponent_y, stream=stream2), d['temp']['d_p_d_y'], stream=stream2) 
        
        #print "Pressure gradient x: {}".format( d['temp']['d_p_d_x'].copy_to_host() )
        #print "Pressure gradient y: {}".format( d['temp']['d_p_d_y'].copy_to_host() )
        
        """Calculate the velocity."""
        d['field']['v_x'] = velocity_with_pml(d['field']['v_x'], d['temp']['d_p_d_x'], d['timestep'], d['density'], d['abs_exp']['x'], d['source']['v']['x'][step], stream=stream1)
        d['field']['v_y'] = velocity_with_pml(d['field']['v_y'], d['temp']['d_p_d_y'], d['timestep'], d['density'], d['abs_exp']['y'], d['source']['v']['y'][step], stream=stream2)
    
    
        stream1.synchronize()
        stream2.synchronize()
        
        """Fourier transform of the velocity."""
        fft(d['field']['v_x'], d['temp']['fft_v_x'], stream=stream1)
        fft(d['field']['v_y'], d['temp']['fft_v_y'], stream=stream2)
        
        stream1.synchronize()
        stream2.synchronize()
        
        
        #print d['temp']['fft_v_y'].copy_to_host()
        #print "Velocity x: {}".format(d['field']['v_x'].copy_to_host())
        #print "Velocity y: {}".format(d['field']['v_y'].copy_to_host())
        
        #print "Source: {}".format(d['source']['p'][step].copy_to_host())
        
        #print "Source: {}".format(d['source']['p'])
        
        
        #print "Velocity exponent y: {}".format(velocity_exponent_y.copy_to_host())
        
        stream1.synchronize()
        stream2.synchronize()
        
        #stream3.synchronize()
        #stream4.synchronize()
        
        velocity_exponent_x = exp(velocity_gradient_exponent(d['k_x'], d['spacing'], stream=stream1), stream=stream1) # This is a constant!!
        velocity_exponent_y = exp(velocity_gradient_exponent(d['k_y'], d['spacing'], stream=stream2), stream=stream2) # This is a constant!!
        
        
        ifft(to_gradient(d['temp']['fft_v_x'], d['k_x'], d['kappa'], velocity_exponent_x, stream=stream1), d['temp']['d_v_d_x'], stream=stream1)
        ifft(to_gradient(d['temp']['fft_v_y'], d['k_y'], d['kappa'], velocity_exponent_y, stream=stream2), d['temp']['d_v_d_y'], stream=stream2)
    
        """And finally the pressure."""
        
        #print len([ d['temp']['p_x'], d['temp']['d_v_d_x'], d['timestep'], d['density'], d['soundspeed'], d['abs_exp']['x'], d['source']['p'][step] ])
        #pressure_with_pml(  d['temp']['p_x'], d['temp']['d_v_d_x'], d['timestep'], d['density'], d['soundspeed'], d['abs_exp']['x'], d['source']['p'][step]  )
        #for i in [ d['temp']['p_x'], d['temp']['d_v_d_x'], d['timestep'], d['density'], d['soundspeed'], d['abs_exp']['x'], d['source']['p'][step] ]:
            #print i , i.shape
            #print i.copy_to_host()
            #try:
                #print i.dtype
            #except AttributeError:
                #print 'None'
        
        stream1.synchronize()
        stream2.synchronize()
        
        #print "Velocity gradient x: {}".format(d['temp']['d_v_d_x'].copy_to_host())
        #print "Velocity gradient y: {}".format(d['temp']['d_v_d_y'].copy_to_host())
        
        #print "Pressure x previous: {}".format(d['temp']['p_x'].copy_to_host())
        #print "Pressure y previous: {}".format(d['temp']['p_y'].copy_to_host())
    
        #print "Abs exp x: {}".format( d['abs_exp']['x'].copy_to_host())
        #print "Abs exp y: {}".format( d['abs_exp']['y'].copy_to_host())
        
        d['temp']['p_x'] = pressure_with_pml(d['temp']['p_x'], d['temp']['d_v_d_x'], d['timestep'], d['density'], d['soundspeed'], d['abs_exp']['x'], d['source']['p'][step], stream=stream1)
        d['temp']['p_y'] = pressure_with_pml(d['temp']['p_y'], d['temp']['d_v_d_y'], d['timestep'], d['density'], d['soundspeed'], d['abs_exp']['y'], d['source']['p'][step], stream=stream2)
    
        stream1.synchronize()
        stream2.synchronize()
        
        #try:
            #print "Source p: {}".format(d['source']['p'][step].copy_to_host())
        #except AttributeError:
            #print "Source p: {}".format(d['source']['p'][step])
            
        #print "Pressure x: {}".format(d['temp']['p_x'].copy_to_host())
        #print "Pressure y: {}".format(d['temp']['p_y'].copy_to_host())
    
        d['field']['p'] = add(d['temp']['p_x'], d['temp']['p_y'], stream=stream3)
        
        #stream3.synchronize()
        #print "Pressure total: {}".format(d['field']['p'].copy_to_host())
        
        
        stream1.synchronize()
        stream2.synchronize()
        stream3.synchronize()
        
        return d
    
    
    def _pre_start(self, data):
        data = super()._pre_start(data)
                
        temp_arrays = ['fft_p', 'fft_v_x', 'fft_v_y', 'd_p_d_x', 'd_p_d_y', 'd_v_d_x', 'd_v_d_y', 'p_x', 'p_y']
        shape = data['field']['p'].shape

        print(data['k_x'].shape)
        
        data['spacing'] *= np.ones(shape, dtype=self.dtype('float'))
        print(data['spacing'].shape)

        for arr in temp_arrays:
            #data['temp'][arr] = cuda.device_array(data['shape'], dtype=self.dtype)
            data['temp'][arr] = np.zeros(shape, dtype=self.dtype('complex'))
        
        return data
    
    def _pre_run(self, data):
        

        #data['temp']['fft_p'] = cuda.device_array(data['shape'], dtype=self.dtype)

        #cuda.select_device(0)
        data = dict_host_to_device(data)   # Make data available on host

        return data
        
        
    def _post_run(self, data):
        
        data = dict_device_to_host(data)   # Make results available on host.
        
        #cuda.close()
        
        super()._post_run(data)
        
        return data
    
