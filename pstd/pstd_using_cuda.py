"""
This module contains a CUDA-accelerated implementation of the k-space PSTD method.
"""

import cmath
import math
import numpy as np

try:
    from numba import autojit
    from numbapro import vectorize, complex64 
    from numbapro import cuda
    from numbapro.cudalib.cufft import fft
    from numbapro.cudalib.cufft import ifft
except ImportError:
    raise ImportWarning("Cannot import numbapro. Cuda not available.") 

import pstd

kappa = vectorize(['complex64(complex64, float64, float64)'], target='cpu')(pstd.kappa)
abs_exp = vectorize(['complex64(complex64, float64)'], target='cpu')(pstd.abs_exp)

pressure_with_pml = vectorize(['complex64(complex64, complex64, float32, float32, float32, complex64, float64)'], target='gpu')(pstd.pressure_with_pml)

velocity_with_pml = vectorize(['complex64(complex64, complex64, float32, float32, complex64, float64)'], target='gpu')(pstd.velocity_with_pml)

#to_pressure_gradient = vectorize(['complex64(complex64, complex64, complex64, float64)'], target='gpu')(pstd.to_pressure_gradient)
#to_velocity_gradient = vectorize(['complex64(complex64, complex64, complex64, float64)'], target='gpu')(pstd.to_velocity_gradient)


#@vectorize(['complex64(complex64, complex64, complex64, float64)'], target='gpu')
#def to_pressure_gradient(pressure_fft, wavenumber, kappa, spacing):
    #return (complex64(1j) * wavenumber * kappa * cmath.exp(complex64(1j)*wavenumber*spacing/2.0) * pressure_fft)#fft2(pressure))


#@vectorize(['complex64(complex64, complex64, complex64, float64)'], target='gpu')
#def to_velocity_gradient(velocity_fft, wavenumber, kappa, spacing):
    #return (complex64(1j) * wavenumber * kappa * cmath.exp(complex64(-1j)*wavenumber*spacing/2.0) * velocity_fft)


@vectorize(["complex64(complex64, complex64)"], target='gpu')
def add(a, b):
    return a + b


@vectorize(["complex64(float32, float64)"], target='gpu')
def pressure_gradient_exponent(wavenumber, spacing):
    """
    Exponent in pressure gradient.
    """
    return complex64(1j) * wavenumber * spacing / complex64(2.0)

@vectorize(["complex64(float32, float64)"], target='gpu')
def velocity_gradient_exponent(wavenumber, spacing):
    """
    Exponent in velocity gradient.
    """
    return complex64(-1j) * wavenumber * spacing / complex64(2.0)

@vectorize(["complex64(complex64, float32, float32, complex64)"], target='gpu')
def to_gradient(transformed, wavenumber, kappa, exponent):
    """
    To Gradient.
    """
    return complex64(1j) * wavenumber * kappa * exponent * transformed

@vectorize(["complex64(complex64)"], target='gpu')
def exp(r):
    """
    Complex exponent.
    """
    x = r.real
    y = r.imag
    return math.exp(x) * (math.cos(y) + complex64(1j) * math.sin(y) )


def dict_host_to_device(d):
    """
    Copy arrays from host to device.
    """
    
    def recurse(d):
        for key, value in d.iteritems():
            if isinstance(value, dict):
                recurse(value)
            else:
                try:
                    d[key] = cuda.to_device(value)
                except TypeError:
                    pass
                except AttributeError:
                    pass
    return recurse(d)
    
def dict_device_to_host(d):
    """
    Copy arrays from device to host.
    """
    def recurse(d):
        for key, value in d.iteritems():
            if isinstance(value, dict):
                recurse(value)
            else:
                try:
                    d[key] = value.copy_to_host()
                except TypeError:
                    pass
                except AttributeError:
                    pass
                
    return recurse(d)
    
 
class PSTD_using_cuda(pstd.PSTD):
    
    @property
    def precision(self):
        """
        Floating point precision.
        
        .. note:: Fixed to single precision.
        """
        return 'single'
    
    
    @staticmethod
    #@autojit
    def update(d):
        
        stream1 = cuda.stream()
        stream2 = cuda.stream()
        stream3 = cuda.stream()
        stream4 = cuda.stream()
        
        step = d['step']
        
        print "Step: {}".format(step)
        
        """Calculate the pressure gradient. Two steps are needed for this."""
        # Calculate FFT of pressure.
        fft(d['field']['p'], d['temp']['fft_p'], stream=stream1)    
        
        stream1.synchronize()
        #print "FFT pressure: {}".format(d['temp']['fft_p'].copy_to_host())
        
        pressure_exponent_x = exp(pressure_gradient_exponent(d['k_x'], d['spacing'], stream=stream1), stream=stream1) # This is a constant!!
        pressure_exponent_y = exp(pressure_gradient_exponent(d['k_y'], d['spacing'], stream=stream2), stream=stream2) # This is a constant!!
        
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
        print "Velocity x: {}".format(d['field']['v_x'].copy_to_host())
        print "Velocity y: {}".format(d['field']['v_y'].copy_to_host())
        
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
        
        print "Velocity gradient x: {}".format(d['temp']['d_v_d_x'].copy_to_host())
        print "Velocity gradient y: {}".format(d['temp']['d_v_d_y'].copy_to_host())
        
        print "Pressure x previous: {}".format(d['temp']['p_x'].copy_to_host())
        print "Pressure y previous: {}".format(d['temp']['p_y'].copy_to_host())
    
        print "Abs exp x: {}".format( d['abs_exp']['x'].copy_to_host())
        print "Abs exp y: {}".format( d['abs_exp']['y'].copy_to_host())
        
        d['temp']['p_x'] = pressure_with_pml(d['temp']['p_x'], d['temp']['d_v_d_x'], d['timestep'], d['density'], d['soundspeed'], d['abs_exp']['x'], d['source']['p'][step], stream=stream1)
        d['temp']['p_y'] = pressure_with_pml(d['temp']['p_y'], d['temp']['d_v_d_y'], d['timestep'], d['density'], d['soundspeed'], d['abs_exp']['y'], d['source']['p'][step], stream=stream2)
    
        stream1.synchronize()
        stream2.synchronize()
        
        try:
            print "Source p: {}".format(d['source']['p'][step].copy_to_host())
        except AttributeError:
            print "Source p: {}".format(d['source']['p'][step])
            
        print "Pressure x: {}".format(d['temp']['p_x'].copy_to_host())
        print "Pressure y: {}".format(d['temp']['p_y'].copy_to_host())
    
        d['field']['p'] = add(d['temp']['p_x'], d['temp']['p_y'], stream=stream3)
        
        stream3.synchronize()
        print "Pressure total: {}".format(d['field']['p'].copy_to_host())
        
        
        stream1.synchronize()
        stream2.synchronize()
        stream3.synchronize()
    
    
    def pre_run(self, data):
        
        super(PSTD_using_cuda, self).pre_run(data)
        
        temp_arrays = ['fft_p', 'fft_v_x', 'fft_v_y', 'd_p_d_x', 'd_p_d_y', 'd_v_d_x', 'd_v_d_y', 'p_x', 'p_y']
        for arr in temp_arrays:
            #data['temp'][arr] = cuda.device_array(data['shape'], dtype=self.dtype)
            data['temp'][arr] = np.zeros(data['shape'], dtype=self.dtype('complex'))
        
        #data['temp']['fft_p'] = cuda.device_array(data['shape'], dtype=self.dtype)
        
        
        #cuda.select_device(0)
        
        dict_host_to_device(data)   # Make data available on host
        
    def post_run(self, data):
        
        dict_device_to_host(data)   # Make results available on host.
        
        #cuda.close()
        
        super(PSTD_using_cuda, self).post_run(data)
    