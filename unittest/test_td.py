import unittest
import numpy as np

from acoustics.td.pstd import PSTD
from acoustics.td.model import Medium, Source, Receiver, Position2D, PML

from numpy.testing import assert_array_almost_equal

from acoustics.td.model import decibel_to_neper, neper_to_decibel, ir2fr

class Utilities(unittest.TestCase):
    
    
    def test_decibel_to_neper(self):
        self.assertEqual(decibel_to_neper(1.0), 0.11512925464970229)
    
    
    def test_neper_to_decibel(self):
        self.assertEqual(neper_to_decibel(1.0), 8.685889638065035)
      
    
    def test_ir2fr(self):
        """
        Test whether the frequency vector is correct.
        """
        
        t = 1.0
        fs = 100.0
        f = 20.0
        ts = np.arange(0, t, 1./fs)
        
        A = 5.0
        
        x = A * np.sin(2. * np.pi * f * ts)
        
        fv, fr = ir2fr(x, fs)
        
        self.assertEqual(fv[np.abs(fr).argmax()], f)

        self.assertAlmostEqual(np.abs(fr).max(), A)

#import acoustics.td.pstd
#import acoustics.td.pstd_using_cuda

#class PSTD(unittest.TestCase):
    
    #def test_pressure_with_pml(self):
        #pass
    
    #def test_velocity_with_pml(self):
        #pass






##class Validate_PSTD(unittest.TestCase):
    ##"""
    ##PSTD related tests.
    ##"""
    
    ##def test_amplitude_and_phase(self):
        ##"""
        ##Validate the response in the frequency domain.
        ##"""
        
        ##x, y = 40.0, 40.0
        
        ##c = 343.2
        
        ##source = Source(Position2D(x/2.0, 5.0), pressure=0.1)
        ##receiver = Receiver(Position2D(x/2.0, 35.0), quantities=['p'])
        ##medium = Medium(soundspeed=c, density=1.296)
        
        ##time = y / c
        
        ##pml = PML((10.0, 10.0), depth=5.0)
        
        ##model = PSTD(time=time, f_max=200.0, pml=pml, medium=medium, cfl=0.3, size=[x, y])
        
        ##model.sources.append(source)
        ##model.receivers.append(receiver)
        
        ##print model.timesteps
        
        ##import logging
        ##logger = logging.getLogger('acoustics.td')    # Module name
        ##logger.setLevel(logging.INFO)                   # Logging level

        
        ##model.run()
        
        
        ##p_pstd = np.fft.fft(receiver.data['p'])
        
        ##print p_pstd
        
        ##r = receiver.position[1] - source.position[1]
        
        ##p_freq = A / r * np.exp(1j*k*r)
        
        
        
    
    

    
        
if __name__ == '__main__':
    unittest.main()
