"""
This module contains an implementation of the k-space PSTD method.
"""
import yaml
import numpy as np
#import scipy.sparse as sp
import abc
import six
import collections
import time
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable

import logging
logger = logging.getLogger(__name__)    # Use module name as logger name

import h5py

try:
    import numba
except ImportError:
    raise ImportWarning("Cannot import numba. JIT not available.")

try:
    from pyfftw.interfaces.numpy_fft import rfft
except ImportError:
    from numpy.fft import rfft

#from sparse import SparseArray # as SparseArray
from sparse_cython.core import SparseArray
#from numpy import ndarray as SparseArray

#from multiprocessing import cpu_count

class SparseList(dict):
    """
    Sparse list.
    
    .. note:: Quick implementation. Better approach would be to encapulse the dictionary.
    
    """
    
    def __init__(self, length, default=0.0):
        
        self._length = length
        self._default = default
    
    def __getitem__(self, i):
        if i >= self._length:
            raise IndexError("Out of range.")
        else:
            try:
                return dict.__getitem__(self, i)
            except KeyError:
                return self._default

    def __len__(self):
        return self._length
    
#def _correct_index(i, s):
    #"""
    #Check whether the index i is within bounds of s. Returns a corrected index.
    #"""
    #if i >= s:
        #raise IndexError("index out of bounds.")
    #elif i < 0:
        #return i + s
    #else:
        #return i
    
def _slice_length(s, l):
    """
    Slicing an array with a slice object results in a (differently sized) array.
    This function calculates the amount of elements in one dimension.
    
    :param s: Slice object
    :param l: Elements in dimensions
    
    """
    return len(range(*s.indices(l)))

#def _sliced_array_shape(slices, shape):
    #"""
    #Slicing an array with a slice object results in a (differently sized) array.
    #This function calculated the shape of the output array.
    
    #:param slices: Tuple of slice objects.
    #:param shape: Shape of input array.
    #"""
    
    #return tuple([_slice_length(s, l) for s, l in zip(slices, shape)] 

class ReverseReference(object):
    """
    The attribute ``label`` this class is assigned to expects an object having ``remote`` attribute.
    
    When instance attribute is updated, remove reference from previous assigned object to instance, and create a reference from the newly assigned object to instance.
    """
    
    def __init__(self, attr, remote):
        self.attr = attr
        self.remote = remote
        
    def __get__(self, instance, owner):
        return instance.__dict__.get(self.attr)
    
    def __set__(self, instance, value):
        try:
            setattr(instance.__dict__.get(self.attr), self.remote, None)
        except AttributeError:
            pass
        instance.__dict__[self.attr] = value
        setattr(value, self.remote, instance) 
    
    
    
@six.add_metaclass(abc.ABCMeta)
class Axes(object):
    """
    Axes
    """

    _model = None
    """
    Reference to model.
    """
    
    DIMENSIONS = []
    
    def __init__(self, shape):
        
        for axis, length in zip(self.DIMENSIONS, shape):
            setattr(self, axis, Axis(self, label=axis, length=length))
            
    def __len__(self):
        return len(self.DIMENSIONS)

    def __getitem__(self, i):
        return getattr(self, self.DIMENSIONS[i])

    def __setitem__(self, i, val):
        setattr(self, self.DIMENSIONS[i], val)

    def __iter__(self):
        for i in self.DIMENSIONS:
            yield getattr(self, i)
    
    @property
    def ndim(self):
        """
        Number of dimensions.
        
        Alias for ``len(self)``.
        """
        return len(self)
    
class Axes2D(Axes):
    """
    Axes for two-dimensional model.
    """
    
    DIMENSIONS = ['x', 'y']

class Axes3D(Axes):
    """
    Axes for three-dimensional model.
    """
    
    DIMENSIONS = ['x'], ['y'], ['z']


#XY = collections.namedtuple('XY', ['x', 'y'])
#XYZ = collections.namedtuple('XYZ', ['x', 'y', 'z'])


class Axis(object):
    """
    Axis of :class:`Model`.
    """

    def __init__(self, axes, label, length=0.0):
        
        self.label = label
        """
        Label indicating which axis this is.
        """
        
        self._axes = axes
        """
        Reference to instance of :class:`Axes`.
        """
        
        self.length_target = length
        """
        Target length of field along this axis.
        """

    @property
    def length_with_pml(self):
        """
        Length of field along this axis when the PML is used.
        """
        return self.nodes_with_pml * self.spacing
    
    @property
    def length_without_pml(self):
        """
        Length of field along this axis when the PML is not used.
        """
        return self.nodes_without_pml * self.spacing
        
    @property
    def length(self):
        """
        Actual length of field along this axis.
        """
        return self.nodes * self.spacing

    @property
    def steps(self):
        """
        Steps
        """
        return np.arange(self.nodes) * self.spacing
    
    @property
    def spacing(self):
        """
        Spacing
        """
        return self._axes._model.grid.spacing
    
    @property
    def nodes_without_pml(self):
        """
        Amount of grid points along this axis when the PML is not used.
        """
        length = self.length_target
        return int(np.ceil(length/self.spacing))    
    
    @property
    def nodes_with_pml(self):
        """
        Amount of grid points along this axis when the PML is used.
        """
        return self.nodes_without_pml + 2 * self._axes._model.pml.nodes
        #length = self.length_target + 2.0 * self._axes._model.pml.depth_target
        #return int(np.ceil(length/self.spacing))
    
    @property    
    def nodes(self):
        """
        Actual amount of grid points along this axis.
        """
        if self._axes._model.pml.is_used:
            return self.nodes_with_pml
        else:
            return self.nodes_without_pml
        #if self._axes._model.pml.is_used:
            #length = self.length_target + 2.0 * self._axes._model.pml.depth_target
        #else:
            #length = self.length_target
        #return int(np.ceil(length/self.spacing))    
        #return int(np.ceil(self.length_target / self.spacing))
    
    ###@property
    ###def nodes_with_pml(self):
        ###"""
        ###Amount of grid points along this axis, including PML nodes.
        ###"""
        ###return self.nodes + 2 * self._axes._model.pml.nodes
    
    
    @property
    def wavenumbers(self):
        """
        Wavenumbers.
        """
        return wavenumbers(self.nodes, self.spacing).astype(self._axes._model.dtype('float'), copy=False)
    
@six.add_metaclass(abc.ABCMeta)
class Position(object):
    """
    Position/Coordinate of :class:`Source` or :class:`Receiver`.
    
    .. note:: Replace with ``Point`` from Geometry module?
    
    """
    
    DIMENSIONS = []
    
    def __iter__(self):
        for i in self.DIMENSIONS:
            yield getattr(self, i)
        #yield self.x
        #yield self.y
        #yield self.z

    def __len__(self):
        return len(self.DIMENSIONS)
    
    def __getitem__(self, i):
        return getattr(self, self.DIMENSIONS[i])
        #if i==0:
            #return self.x
        #elif i==1:
            #return self.y
        #elif i==2:
            #return self.z
        #else:
            #raise IndexError()

    def __setitem__(self, i, val):
        setattr(self, self.DIMENSIONS[i], val)
        #if i==0:
            #self.x = val
        #elif i==1:
            #self.y = val
        #elif i==2:
            #self.z = val
        #else:
            #raise IndexError()

class Position2D(Position):
    """
    Position in 2D.
    """
    
    DIMENSIONS = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        """
        x
        """
        self.y = y
        """
        y
        """

    def __str__(self):
        return "Position2D({:.2f}, {:.2f})".format(self.x, self.y)


###class Position3D(Position):
    ###"""
    ###Position in 3D.
    ###"""
    
    ###DIMENSIONS = ['x', 'y', 'z']
    
    ###def __init__(self, x, y, z):
        ###self.x = x
        ###"""
        ###x
        ###"""
        ###self.y = y
        ###"""
        ###y
        ###"""
        ###self.z = z
        ###"""
        ###z
        ###"""

    ###def __str__(self):
        ###return "Position3D({}, {}, {})".format(self.x, self.y, self.z)


@six.add_metaclass(abc.ABCMeta)
class Transducer(object):
    
    _model = None
    
    def __init__(self, position):
        
        
        self.position_target = position
        """
        Position of the source. See :class:`Position`.
        """
    @property
    def position(self):
        """
        Actual position, taking into account the PML.
        The coordinates are shifted along each axis by the depth of the PML.
        """
        if self._model.pml.is_used:
            return self.position_with_pml
        else:
            return self.position_without_pml
        
        
    @property
    def position_with_pml(self):
        """
        Actual position, taking into account the PML.
        """
        spacing = self._model.grid.spacing
        return Position2D(*(x*spacing for x in self.position_in_nodes_with_pml))
        
    @property
    def position_without_pml(self):
        """
        Actual position, without PML.
        """
        spacing = self._model.grid.spacing
        return Position2D(*(x*spacing for x in self.position_in_nodes_without_pml))
    
    @property
    def position_in_nodes(self):
        """
        Actual position in nodes.
        """
        if self._model.pml.is_used:
            return self.position_in_nodes_with_pml
        else:
            return self.position_in_nodes_without_pml
        #pml = self._model.pml.nodes
        #spacing = self._model.grid.spacing
        #return tuple(int(round(x/spacing))+pml for x in self.position_target)
       
    @property
    def position_in_nodes_without_pml(self):
        """
        Actual position in nodes without PML. 
        """
        spacing = self._model.grid.spacing
        return tuple(int(round(x/spacing)) for x in self.position_target)
       
    
    @property
    def position_in_nodes_with_pml(self):
        """
        Actual position in nodes with PML.
        """
        pml = self._model.pml.nodes
        
        return tuple(x+pml for x in self.position_in_nodes_without_pml)
        
        #return tuple(int(round(x/spacing))+pml for x in self.position_target)
       
    
    
def mass(pressure, c, spacing, ndim):
    """
    Mass.
    
    :param pressure: Sound pressure
    :param c: Speed of sound
    :param spacing: Spacing
    :param ndim: Amount of dimensions.
    
    .. math:: S_{M_i} = p \\frac{2 }{c N \\Delta x}
    
    where :math:`i` represents the axis.
    """
    return pressure * 2.0 / (c * ndim * spacing)

def force(velocity, c, spacing):
    """
    Force.
    
    :param velocity: Particle velocity.
    :param c: Speed of sound.
    :param spacing: Spacing
    
    .. math:: S_{F_i} = u_i \\frac{2 c}{\\Delta x}
    
    where :math:`i` represents the axis.
    """
    return velocity * 2.0 * c / spacing
  

@six.add_metaclass(abc.ABCMeta)
class Source(object):
    """
    Source
    """
    
    def __init__(self, quantity, component=None):
        
        self.quantity = quantity
        """
        Quantity of source. Pressure or velocity.
        """
        
        if self.quantity == 'velocity':
            if self.component:
                self.component = component
            else:
                raise ValueError("Need argument component for velocity source.")
        

#class FieldSource(object):
    #"""
    #Custom source field.
    #"""
    
    #def __init__(self, quantity, field, component=None):
        
        #super(Field, self).__init__(self, quantity, field)

        #self.field = field


class PointSource(Source, Transducer):
    """
    Point source.
    """
    
    def __init__(self, position, quantity, component=None, excitation='pulse', amplitude=None, frequency=None):
        
        Transducer.__init__(self, position)
        Source.__init__(self, quantity, component)

        self.excitation = excitation
        
        self.amplitude = amplitude
        
        self.component = component
        
        
        
        #if excitation=='sine':
        self.frequency = frequency

    #@field.setter
    #def field(self, field):
        #self._field = field
        #self.excitation = 'custom'
    
    #@field.deleter
    #def field(self):
        #self._field = None
        #self.excitation = None
        
    @property
    def field(self):
        """
        Field.
        """
        
        if self.excitation=='sine':
            return self._field_sine()
        elif self.excitation=='pulse':
            return self._field_pulse()
        elif self.excitation=='custom':
            return self._field_custom()
        else:
            raise ValueError("Excitation type is not specified.")

    #@property
    #def _shape(self):
        #shape = list(self._model.grid.shape)
        #shape.insert(0, self._model.timesteps)
        #return tuple(shape)
        
    def _converted_amplitude(self):
        if self.quantity=='pressure':
            return self.mass
        elif self.quantity=='velocity':
            return self.force
        else:
            raise ValueError("Incorrect quantity.")
        
    def _field_sine(self):
        """
        Field in space and time due to sine wave excitation.
        """
        if not self.frequency:
            raise ValueError("Frequency of sine source is not specified.")

        model = self._model
        shape = model.shape_spacetime
        timestep = model.timestep
        timesteps = model.timesteps
        position = self.position_in_nodes
        
        field = SparseArray(shape, dtype=model.dtype('float'))
        
        times = np.arange(timesteps) * timestep
        
        amplitude = self._converted_amplitude()
        
        signal = amplitude * np.sin(2.0 * np.pi * self.frequency * times)
        #print len(signal)
        #field[:,position] = signal
        # Workaround since SparseArray does not support assigning a sequence.
        #print position[0], position[1]
        for i, v in enumerate(signal):
            #print i, v
            field[(i, position[0], position[1])] = v
        #print field._data
        return field
        
    def _field_custom(self):
        """
        Custom emission signal.
        """
        signal = self._converted_amplitude
        
        if len(signal) != self._model.timesteps:
            raise ValueError("Signal length does not match the model timesteps.")
        
        field = SparseArray(self._model.shape_spacetime, dtype=model.dtype('float'))
        field[:, position] = signal
        return field
    
    def _field_pulse(self):
        """
        Field in space and time due to sine wave excitation.
        """
        model = self._model
        shape = model.shape_spacetime
        timestep = model.timestep
        timesteps = model.timesteps
        position = self.position_in_nodes
        amplitude = self._converted_amplitude()
        
        #field = SparseArray(shape, dtype=model.dtype('float'))
        field = SparseList(length=timesteps, default=0.0)
        
        spacing = self._model.grid.spacing      # Grid spacing
        
        # We now calculate a small pulse, and then fit it in
        pulse = self.gaussian_pulse(amplitude, spacing)  # Small pulse
        shape = pulse.shape
        offset = tuple(int(round(i/2.0)) for i in shape)
        position = self.position_in_nodes
        
        #pulse_grid = SparseArray(shape=model.grid.shape, dtype=model.dtype('float'))
        pulse_grid = np.zeros(self._model.grid.shape, dtype=self._model.dtype('float'))   # Grid
        pulse_grid[ position[0]-offset[0]: position[0]-offset[0]+shape[0], position[1]-offset[1] : position[1]-offset[1]+shape[1] ] = pulse
        
        #field[0,:,:] = pulse_grid # 
        # Workaround since SparseArray does not support assigning a sequence.
        #for i, v in np.ndenumerate(pulse_grid):
            #field[0, i[0], i[1]] = v
        field[0] = pulse_grid
        return field
        
    #@staticmethod
    #def gaussian_pulse(position, amplitude, spacing, shape=(16,16), a=0.3):
        #"""
        #Create initial pulse.
        
        #:param position: Iterator containing indices of source position.
        #:param amplitude: Energy in the pulse. :math:`A`.
        #:param shape: Shape of the grid.
        #:param spacing: Grid spacing.
        
        #.. math:: g = a \\left( \\frac{1}{\\Delta x} \\right)^2
        
        #.. math:: p = A e^{ -g \\sum_{i=0}^n \\left(x_i - x_{p,i}\\right)^2  }
        
        #.. note:: Cut of the exponential. Then the source terms can be stored as sparse arrays.
        
        #"""
        #g = a * (1.0/spacing)**2.0
        #vectors = [np.arange(1, n+1) * spacing for n in shape]
        #grids = np.meshgrid(*vectors, indexing='ij')
        #pulse = amplitude * np.exp( -g * sum([(grid - pos)**2.0 for grid, pos in zip(grids, position)]) )
        #return pulse
       
    @staticmethod
    def gaussian_pulse(amplitude, spacing, shape=(15,15), a=0.3):
        """
        Create initial pulse.
        
        :param amplitude: Energy in the pulse. :math:`A`.
        :param spacing: Grid spacing.
        :param shape: Shape of the grid.
        :param a: Sharpness
        
        .. math:: g = a \\left( \\frac{1}{\\Delta x} \\right)^2
        
        .. math:: p = A e^{ -g \\sum_{i=0}^n \\left(x_i - x_{p,i}\\right)^2  }
        
        """
        position = tuple( spacing * int(round(i/2.0)) for i in shape)
        
        g = a * (1.0/spacing)**2.0
        vectors = [np.arange(1, n+1) * spacing for n in shape]
        grids = np.meshgrid(*vectors, indexing='ij')
        pulse = amplitude * np.exp( -g * sum([(grid - pos)**2.0 for grid, pos in zip(grids, position)]) )
        return pulse
       
       
       
    @property
    def mass(self):
        """
        Mass contribution.
        
        See :func:`mass`.
        """
        if self.quantity == 'pressure':
            m = mass(self.amplitude, self._model.medium.soundspeed_mean, self._model.grid.spacing, self._model.ndim)
            try:
                return float(m)
            except TypeError:
                return m
    
    @property  
    def force(self):
        """
        Force contribution. Returns a tuple where each element represents the component of the force in the respective dimension.
        
        See :func:`force`.
        """
        if self.quantity == 'velocity':
            f = force(self.amplitude, self._model.medium.soundspeed_mean, self._model.grid.spacing)
            try:
                return float(f)
            except TypeError:
                return f
            
    def _attributes(self):

        source_fields = ['position', 'position_with_pml', 'position_without_pml', 
                         'position_in_nodes', 'position_in_nodes_with_pml', 'position_in_nodes_without_pml',
                         'quantity', 'component', 'excitation', 'amplitude', 
                         'frequency', 'mass', 'force']        
    
        return { item : getattr(self, item) for item in source_fields}
        
#class Excitation(object):
    #pass


#@six.add_metaclass(abc.ABCMeta)
#class Source(Transducer):
    #"""
    #Source
    #"""
    
    #def __init__(self, position, amplitude, excitation=None):
        
        #super(Source, self).__init__(position)
        
        #excitation = excitation if excitation else Pulse()
        
    #@abc.abstractmethod
    #def field(self):
        #pass



#class Pressure(Source):
    #"""
    #Pressure source.
    #"""
    
    #@property
    #def mass(self):
        #"""
        #Mass contribution.
        
        #See :func:`mass`.
        #"""
        #return mass(self.amplitude, self._model.medium.soundspeed_mean, self._model.grid.spacing, self._model.ndim)
    
    
    #def field(self):
        
        #return self.excitation._field(self)
    
#class Velocity(Source):
    #"""
    #Velocity source.
    #"""
    
    #def __init__(self, position, amplitude, excitation=None, component=None):
        
        #super(Velocity, self).__init__(self, 
    
    #@property  
    #def force(self):
        #"""
        #Force contribution. Returns a tuple where each element represents the component of the force in the respective dimension.
        
        #See :func:`force`.
        #"""
        #return tuple(force(velocity, self._model.medium.soundspeed_mean, self._model.grid.spacing) for velocity in self.amplitude)
        

    #def field(self):
        
        #return tuple(self.excitation._field(self) for force_component in self.force)


#@six.add_metaclass(abc.ABCMeta)
#class Excitation(object):
    #"""
    #Excitation.
    #"""
    
    #def __init__(self):
        #pass

#class Pulse(Excitation):
    #"""
    #Pulse excitation.
    #"""
    
    #def __init__(self):
        #super(Pulse, self).__init__()
    
    #def _initial_pulse(self, amplitude):
        #"""
        #Calculate the initial pulse.
        #"""
        
        #spacing = self._model.grid.spacing      # Grid spacing
        #pulse = self.gaussian_pulse(amplitude, spacing)  # Pulse
        #shape = pulse.shape
        #offset = tuple(int(round(i/2.0)) for i in shape)
        #position = self.position_in_nodes
        
        #grid = np.zeros(self._model.grid.shape, dtype=self._model.dtype('float'))   # Grid
        #grid[ position[0]-offset[0]: position[0]-offset[0]+shape[0], position[1]-offset[1] : position[1]-offset[1]+shape[1] ] = pulse
        
        #return grid

    #def _field(self, amplitude, timestep, timesteps):
        #pass
    
    
    #@staticmethod
    #def gaussian_pulse(amplitude, spacing, shape=(15,15), a=0.3):
        #"""
        #Create initial pulse.
        
        #:param amplitude: Energy in the pulse. :math:`A`.
        #:param spacing: Grid spacing.
        #:param shape: Shape of the grid.
        #:param a: Sharpness
        
        #.. math:: g = a \\left( \\frac{1}{\\Delta x} \\right)^2
        
        #.. math:: p = A e^{ -g \\sum_{i=0}^n \\left(x_i - x_{p,i}\\right)^2  }
        
        #"""
        #position = tuple( spacing * int(round(i/2.0)) for i in shape)
        
        #g = a * (1.0/spacing)**2.0
        #vectors = [np.arange(1, n+1) * spacing for n in shape]
        #grids = np.meshgrid(*vectors, indexing='ij')
        #pulse = amplitude * np.exp( -g * sum([(grid - pos)**2.0 for grid, pos in zip(grids, position)]) )
        #return pulse

#class Sine(Excitation):
    #"""
    #Sine excitation.
    #"""
    
    #def __init__(self, amplitude, frequency):
        #super(Pulse, self).__init__()
        #self.frequency = frequency
        #"""
        #Frequency of the sine excitation.
        #"""
    
    #def _signal(self, times):
        #return self.amplitude * np.sin(2.0 * np.pi * self.frequency * times)
    
    #def _field(self, source):
        
        #amplitude = self.amplitude
        #model = source._model
        
        #timestep = model.timestep
        #timesteps = model.timesteps
        #grid_shape = model.grid.shape
        #position = source.position_in_nodes
        
        #shape = tuple(list(grid_shape).insert(0, timesteps))
        #field = SparseArray(shape, dtype=model.dtype('float'))
        
        #times = np.arange(timesteps, timestep)
        #signal = source.amplitude * np.sin(2.0 * np.pi * self.frequency * times)
        
        #field[:,position] = signal
        
        #return field
    
#class Custom(Excitation):
    
    #def __init__(self):
        #super(Pulse, self).__init__()
        
    
    #def _field(self, source):
        #pass

class ReceiverArray(object):
    """
    Receiver array.
    """
    
    def __init__(self, quantities=None, last_value_only=False, filename=None):
        
        self.quantities = quantities if quantities else ['p']
        """
        List of quantities to record.
        """
        self.last_value_only = last_value_only
        self.filename = filename
        
    def shape(self):
        pass
    
    def shape_with_pml(self):
        pass
    
    def shape_without_pml(self):
        pass
    
    def size(self):
        pass
    
    def size_with_pml(self):
        pass
    
    def size_without_pml(self):
        pass
    
    
    
    

class Receiver(Transducer):
    """
    Receiver.
    """
    
    _position_in_nodes = None
    """
    Position in nodes. Static value that is updated during pre-run.
    """
    
    def __init__(self, position, quantities=None, last_value_only=False, filename=None):
        
        super(Receiver, self).__init__(position)
        
        self.last_value_only = last_value_only
        """
        Record only the final value (True) or the impulse response (False).
        """
        
        self.quantities = quantities if quantities else ['p']
        """
        List of quantities to record.
        """
        
        #self.filename = filename
        #"""
        #Store in file.
        #"""
        
        self.data = dict()
        """
        "Recorded" data.
        """
        
        #self.store = store
        #"""
        #Store data.
        #"""

    ##def store(self, data):
        ##"""
        ##Select and store data.
        ##"""
        ##for q in quantities:
            ##self.data[q] = data['field'][q]

    
    def _attributes(self):
        receiver_fields = ['position', 'position_with_pml', 'position_without_pml', 
                         'position_in_nodes', 'position_in_nodes_with_pml', 'position_in_nodes_without_pml',
                         'quantities', 'last_value_only']
        
        return { item : getattr(self, item) for item in receiver_fields}
                
    


class Medium(object):
    """
    Medium. 
    
    See also :class:`acoustics.atmosphere.Atmosphere`.
    """
    
    def __init__(self, soundspeed, density):#, refractive_index=1.0):
        
        self.soundspeed = soundspeed
        """
        Speed of sound :math:`c_0`.
        
        .. note:: In case of a inhomogeneous atmosphere, this value is an array.
        
        """
        self.density = density
        """
        Density :math:`\\rho`.
        """    
    
    @property
    def soundspeed_mean(self):
        """
        Mean soundspeed.
        """
        return np.mean(self.soundspeed)
    
    @property
    def soundspeed_for_calculation(self):
        """
        Soundspeed that, if required, takes into account the PML.
        
        .. note::   In case :attr:`soundspeed` a single value is, this property returns the same value.
                    Else, the array is expanded to take into account the increased grid size due to the PML.
                    The mean soundspeed is used for panning.
        
        .. warning:: 2D only.
        
        """
        try:
            n = len(self.soundspeed)
        except TypeError:   # Single
            return self.soundspeed
        
        if len(self.soundspeed) == 1 or self.soundspeed.shape == self._model.grid.shape: # Generally the case we have no PML.
            return self.soundspeed
        else:
            if not self.soundspeed.shape==self._model.grid.shape_without_pml:
                raise ValueError("Soundspeed has dimensions {} while a single value or shape {} is required.".format(self.soundspeed.shape, self._model.grid.shape_without_pml))
            
            depth = self._model.pml.nodes
            c = self.soundspeed_mean
            grid = np.ones(self._model.grid.shape, dtype=self._model.dtype('float')) * c
            #grid[depth:-depth, depth:-depth] = self.soundspeed
            shape = self.soundspeed.shape
            grid[depth:depth+shape[0], depth:depth+shape[1]] = self.soundspeed
            
            return grid

        

#def requires_object(obj_name):
    """
    Catch attribute error and mention obj_name.
    """
    #def real_decorator(function):
        #def wrapper(*args, **kwargs):
            
            #try:
                #function(*args, **kwargs)
            #except AttributeError:
                #raise AttributeError("Object has not yet been assigned to an instance of {}".format(obj_name))
        #return wrapper
    #return real_decorator

class PML(object):
    """
    Perfectly Matched Layer.
    
    """
    
    _model = None
    """
    Reference to instance of :class:`Model`.
    """
    
    depth_target = None
    """
    Target depth or length of the PML.
    """
    
    def __init__(self, absorption_coefficient, depth):
        
        self.absorption_coefficient = absorption_coefficient
        """
        Maximum absorption coefficient :math:`\\alpha`. Tuple.
        """
        self.depth_target = depth

    @property
    def is_used(self):
        """
        Whether the PML is used or not.
        
        This is a shortcut for ``model.settings['pml']['use']``.
        """
        return self._model.settings['pml']['use']

    @property
    def nodes(self):
        """
        Amount of nodes corresponding to the depth of this PML.
        """
        return int(np.ceil(self.depth_target / self._model.grid.spacing))

    @property
    def depth(self):
        """
        Actual depth of the PML. The actual depth is equal or slightly higher than the given depth due to discretization.
        
        """
        return (self._model.grid.spacing * self.nodes)
        
    def generate_grid(self):
        """
        Generate the PML grids. A dictionary is returned in which each item represents the PML for a dimension.
        
        """
        shape = self._model.grid.shape
        
        pml = dict()
        
        pml['x'] = np.zeros(shape, dtype=self._model.dtype('float'))
        pml['y'] = np.zeros(shape, dtype=self._model.dtype('float'))
        
        if not self._model.settings['pml']['use']:  # Return arrays with zeros
            return pml
        
        depth = self.depth
        nodes = self.nodes   # Depth of PML in nodes.
        
        if nodes <=1:
            raise ValueError("Increase size of PML.")
        
        """Create a PML sparse matrix for each dimensions."""
        #pml['x'] = sp.lil_matrix(shape, dtype=self._model.dtype)
        #pml['y'] = sp.lil_matrix(shape, dtype=self._model.dtype)
        
        pml['y'][:,:+nodes] = np.tile(self.absorption_coefficient_grid(np.linspace(-depth, 0, nodes), 0.0, -depth, alpha_max=self.absorption_coefficient[1]), (shape[0],1)) # Top
        pml['y'][:,-nodes:] = np.tile(self.absorption_coefficient_grid(np.linspace(0, +depth, nodes), 0.0, +depth, alpha_max=self.absorption_coefficient[1]), (shape[0],1)) # Bottom
        
        pml['x'][:+nodes,:] = np.tile(self.absorption_coefficient_grid(np.linspace(-depth, 0, nodes), 0.0, -depth, alpha_max=self.absorption_coefficient[0]), (shape[1],1)).T # Left
        pml['x'][-nodes:,:] = np.tile(self.absorption_coefficient_grid(np.linspace(0, +depth, nodes), 0.0, +depth, alpha_max=self.absorption_coefficient[0]), (shape[1],1)).T # Right
       
        #pml['x'] = pml['x'].tocsc()
        #pml['y'] = pml['y'].tocsc()
        
        return pml
        
    
    @staticmethod
    def absorption_coefficient_grid(x, x_0, depth, alpha_max, m=4.0):
        """
        Calculate the Perfectly Matched Layer absorption coefficient in dB/s or Np/s for every node within a PML.
        
        :param x: Position in the PML :math:`x`
        :param x_0: :math:`x_0`
        :param x_max: :math:`x_max`
        :param depth: Depth of the PML :math:`D`
        :param m: :math:`m`
        
        .. math:: \\alpha_{\\xi} = \\alpha_{max} \\left( \\frac{\\xi - \\xi_0}{D} \\right) ^ m
        
        """
        #return 50.0
        return alpha_max * ((x - x_0) / (depth*1.1))**m   # 1.01 to prevent division by zero
    
    def plot(self, filename=None):
        
        pml = self.generate_grid()
        
        fig = plt.figure(figsize=(16,12), dpi=80)
        
        x = np.arange(self._model.axes.x.nodes+1) * self._model.grid.spacing
        y = np.arange(self._model.axes.y.nodes+1) * self._model.grid.spacing
        xl = self._model.axes.x.length
        yl = self._model.axes.y.length
        
        fig = plt.figure(figsize=(16, 12), dpi=80)
        grid = AxesGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.3, cbar_mode="single")
        
        ax1 = grid[0]#.add_subplot(131, aspect='equal')
        ax1.set_title(r"$x$-direction")
        plot1 = ax1.pcolormesh(x, y, pml['x'].T)
        ax1.set_xlim(0.0, xl)
        ax1.set_ylim(0.0, yl)
        ax1.grid()
        grid.cbar_axes[0].colorbar(plot1)
        
        ax2 = grid[1]#fig.add_subplot(132, aspect='equal')
        ax2.set_title(r"$y$-direction")
        plot2 = ax2.pcolormesh(x, y, pml['y'].T)
        ax2.set_xlim(0.0, xl)
        ax2.set_ylim(0.0, yl)
        ax2.grid()
        grid.cbar_axes[0].colorbar(plot2)
        
        if filename:
            fig.savefig(filename)
        else:
            return fig
    
    def _attributes(self):
        pml_fields = ['is_used', 'nodes', 'depth']
        return {item : getattr(self, item) for item in pml_fields }
        
        
    
    #def plot_absorption_coefficient(self):
        #pass
    
    
class Grid(object):
    """
    Grid for TD model.
    """
    
    _model = None
    """
    Reference to instance of :class:`Model`.
    """
    
    _spacing = None
    
    def __init__(self, spacing=None):
        
        self.spacing = spacing
    
    @property
    def size(self):
        """
        Size of the model in meters.
        """
        return tuple(axis.length for axis in self._model.axes)

    @property
    def size_with_pml(self):
        """
        Size of the model in meters when the PML is used.
        """
        return tuple(axis.length_with_pml for axis in self._model.axes)
    
    @property
    def size_without_pml(self):
        """
        Size of the model in meters when the PML is not used.
        """
        return tuple(axis.length_without_pml for axis in self._model.axes)
    
    @property
    def shape(self):
        """
        Shape of the grid.
        """
        return tuple([axis.nodes for axis in self._model.axes])
    
    @property
    def shape_with_pml(self):
        """
        Shape of the grid when the PML is used.
        """
        return tuple([axis.nodes_with_pml for axis in self._model.axes])
    
    @property
    def shape_without_pml(self):
        """
        Shape of the grid when the PML is not used.
        """
        return tuple([axis.nodes_without_pml for axis in self._model.axes])
    
    @property
    def spacing(self):
        """
        Spacing of the grid.
        
        .. math:: \\Delta x = \\frac{c_{min}}{2 f_{max}}
        """
        if self._spacing:
            return self._spacing
        else:
            return np.min(self._model.medium.soundspeed) / self.spatial_sample_frequency #(2.0 * self._model.f_max)
            
    @spacing.setter
    def spacing(self, x):
        self._spacing = x
    
    @spacing.deleter
    def spacing(self):
        self._spacing = None
    
    @property
    def spatial_sample_frequency(self):
        """
        Spatial sample frequency.
        
        .. math:: f_{s, spatial} = 2 f_{max}
        """
        return 2.0 * self._model.f_max
    
    def _attributes(self):
        grid_fields = ['size', 'size_with_pml', 'size_without_pml', 'shape',
                'shape_with_pml', 'shape_without_pml', 'spacing', 
                'spatial_sample_frequency']

        return {item : getattr(self, item) for item in grid_fields }
        
        
@six.add_metaclass(abc.ABCMeta)
class Model(object):
    """
    Abstract model for Time-Domain simulations.
    """
    
    precision = 'double'
    """
    Floating point precision. Valid values are 'single' and 'double' for respectively single and double precision.
    """
    
    _file_handler = None
    """
    HDF5 file handler.
    """
    
    def dtype(self, sort):
        """
        dtype of the array for given ``sort``.
        """
        items = {'single' : 32,
                 'double' : 64,
                 }
        bits = items[self.precision]
        
        if sort=='complex':
            return np.dtype(sort + str(2*bits))
        else:
            return np.dtype(sort + str(bits))
    
    #_f_max = None
    
    _timesteps = None

    axes = None
    """
    Axes in the model. See :class:`Axes` and :class:`Axis`.
    """
        
    grid = ReverseReference(attr='grid', remote='_model')
    """
    Grid. See :class:`Grid`.
    """
    pml = ReverseReference(attr='pml', remote='_model')
    """
    Perfectly Matched Layer. See :class:`PML`.
    """
    axes = ReverseReference(attr='axes', remote='_model')
    """
    Axes. See :class:`Axes`.
    """
    medium = ReverseReference(attr='medium', remote='_model')
    
    
    ###_threads = None
    
    ###@property
    ###def threads(self):
        ###"""
        ###Amount of threads to use when multithreading is set to True.
        ###"""
        ###if self.multithreading:
            ###if self._threads:
                ###return self._threads
            ###else:
                ###return cpu_count
        ###else:
            ###return 1
    
    ###@threads.setter
    ###def threads(self, x):
        ###self._threads = x
    
    def __init__(self, time, f_max, medium, pml, cfl=0.05, spacing=None, axes=None, size=None, settings=None, grid=None):
        
        
        super(Model, self).__init__()
        
        self.settings = DEFAULT_SETTINGS
        """
        Settings.
        """
        if settings:
            self.settings.update(settings) # Apply given settings.
        
        #self.multithreading = True
        #"""
        #Allow multithreading.
        #"""
        
        
        #self.axes = Axes2D(*[Axis(self, length=i) if i else None for i in size if i]) # Create an axis for each dimension that is not None
        if size:
            if len(size)==2:
                self.axes = Axes2D(shape=size)
            elif len(size)==3:
                self.axes = Axes3D(shape=size)
        elif axes:
            if isinstance(axes, Axes):
                self.axes = axes
            else:
                TypeError("Wrong type axes.")
        else:
            raise RuntimeError("Either axes or size should be specified.")
        
        self.grid = grid if grid else Grid(spacing)
        """
        Grid. See :class:`Grid`.
        """
        
        
        self.f_max = f_max
        """
        Upper frequency limit of the model. Above this frequency limit the model is not reliable.
        """
        
        self.medium = medium
        """
        Medium. See :class:`Medium`.
        """
        
        self.time_target = time
        """
        Target simulation time.
        """
        
        self.cfl = cfl
        
        self.pml = pml# if pml else PML()

        
        self.sources = list()
        """
        A list of sources. See :class:`Source`.
        """
        
        self.receivers = list()
        """
        A list of receivers. See :class:`Receiver`.
        """

    @property
    def cfl(self):
        """
        Courant-Friedrichs-Lewy number. The CFL number can be calculated using :func:`CFL`.
        """
        return self._cfl

    @cfl.setter
    def cfl(self, x):
        if not x <= np.mean(self.medium.soundspeed)/np.max(self.medium.soundspeed):
            raise ValueError("CFL too high.")
        self._cfl = x
        
    @property
    def constant_field(self):
        """
        Constant refractive-index field.
        """
        if self.medium.refractive_index.ndim > 2:
            return False
        else:
            return True
    
    @property
    def ndim(self):
        """
        Amount of dimensions. This value is determined from the amount of axes.
        """
        return len(self.axes)

    
    _timestep = None

    @property
    def timestep(self):
        """
        Timestep.
        
        .. math:: \\Delta t = \\frac{ \\mathrm{CFL} \\Delta x } {c_{max}}
        
        """
        if self._timestep:
            return self._timestep
        else:
            return self.cfl * self.grid.spacing / float(np.max(self.medium.soundspeed))
        
    @timestep.setter
    def timestep(self, x):
        self._timestep = x
        
    @property    
    def timesteps(self):
        """
        Amount of timesteps to be taken. That value can be overridden.
        
        .. math:: n = \\frac{t}{\\Delta t}
        """
        if self._timesteps:
            return self._timesteps
        else:
            return int(np.ceil(self.time_target / self.timestep))
    
    @timesteps.setter
    def timesteps(self, x):
        self._timesteps = x 
    
    @timesteps.deleter
    def timesteps(self, x):
        self._timesteps = None
    
    @property
    def time(self):
        """
        Total simulation time.
        """
        return self.timesteps * self.timestep
    
    @time.setter
    def time(self, x):
        self.time_target = x
    
    @property
    def sample_frequency(self):
        """
        Temporal sample frequency in Hz. Inverse of :attr:`timestep`.
        """
        return 1.0 / self.timestep
    
    @property
    def shape_spacetime(self):
        
        shape = list(self.grid.shape)
        shape.insert(0, self.timesteps)
        return shape
    
    @property
    def shape_spacetime_with_pml(self):
        
        shape = list(self.grid.shape_with_pml)
        shape.insert(0, self.timesteps)
        return shape
        
    @property
    def shape_spacetime_without_pml(self):
        
        shape = list(self.grid.shape_without_pml)
        shape.insert(0, self.timesteps)
        return shape
        
    #@property
    #def _frequencies(self):
        #"""
        #Frequencies up to highest accurate frequency.
        #"""
        #return frequencies(self.nodes, self.sample_frequency).astype(self._axes._model.dtype('float'), copy=False)
    
    def _prepare_transducers(self):
        """
        Prepare transducers. Adds reference to model.
        """
        for source in self.sources:
            source._model = self
        for receiver in self.receivers:
            receiver._model = self
        
    
    def _prepare_recorder(self):
        """
        Prepare recorder.
        """
        
        if self.settings['recording']['use']:
            f = h5py.File(self.settings['recording']['filename'],'a')
            
            if self.settings['recording']['groupname']:
                groupname = self.settings['recording']['groupname']
            else:
                groupname = datetime.datetime.utcnow().replace(microsecond=0).isoformat()
            
            group = f.create_group(groupname)
            
            self._file_handler = (f, group)
            
            #shape = self.shape_spacetime
            
            shape = list()
            for d, length in zip(self.settings['recording']['slice'], self.grid.shape):
                if isinstance(d, int):
                    shape.append(1)
                elif isinstance(d, slice):
                    shape.append(_slice_length(d, length))
                    
            #shape = [i for i in self.settings['recording']['slice']]
            shape.insert(0, self.timesteps)
            
            shape = tuple(shape)
            #shape = (self.timesteps, 
                     #self.settings['recording']['slice'][0],
                     #self.settings['recording']['slice'][1])

            field = group.create_group('field')
            for quantity in self.settings['recording']['quantities']:
                field.create_dataset(quantity, shape, dtype=self.dtype('float'))

            if self.settings['recording']['meta']:
                # Main parameters
                
                model = group.create_group['model']
                for key, value in self._attributes():
                    model.attrs[key] = value
                    
                #group.attrs.create('cfl', self.cfl)
                #group.attrs.create('ndim', self.ndim)
                #group.attrs.create('timestep', self.timestep)
                #group.attrs.create('timesteps', self.timesteps)
                #group.attrs.create('time', self.time)
                #group.attrs.create('temporal_sample_frequency', self.sample_frequency)
                
                # Grid parameters
                grid = group.create_group('grid')
                for key, value in self.grid._attributes():
                    grid.attrs[key] = value
                
                # PML parameters
                pml = group.create_group('pml')
                for key, value in self.pml._attributes():
                    pml.attrs[key] = value
                
                
                #grid.attrs.create('spacing', self.grid.spacing)
                #grid.attrs.create('spatial_sample_frequency', self.grid.spatial_sample_frequency)
                #grid.attrs.create('size', self.grid.size)
                #grid.attrs.create('size_with_pml', self.grid.size_with_pml)
                #grid.attrs.create('size_without_pml', self.grid.size_without_pml)
                #grid.attrs.create('shape', self.grid.shape)
                #grid.attrs.create('shape_with_pml', self.grid.shape_with_pml)
                #grid.attrs.create('shape_without_pml', self.grid.shape_without_pml)
                
                # Medium parameters
                medium = group.create_group('medium')
                medium.create_dataset('soundspeed', data=self.medium.soundspeed)
                medium.attrs.create('density', self.medium.density)
                medium.attrs.create('soundspeed_mean', self.medium.soundspeed_mean)
                medium.create_dataset('soundspeed_for_calculation', data=self.medium.soundspeed_for_calculation)
                s = self.settings['recording']['slice']
                
                
                
                
                
                #try:
                    #c = group.create_dataset('c', data=self.medium.soundspeed[s])
                #except IndexError:
                    #c = group.create_dataset('c', data=self.medium.soundspeed)

                # PML parameters
                #pml = group.create_group('pml')
                #pml.attrs.create('is_used', self.pml.is_used)
                #pml.attrs.create(
            

    #@autojit
    def _pre_run(self, data):
        """
        This function is performed before the simulation begins.
        
        
        The following actions are performed:
        
        * Create PML using :meth:`PML.generate_grid' if PML is used.
        * Create field arrays.
        * Create source arrays.
        * Start timer for progress logging.
        
        """
        logger.info("Progress: Starting simulation")
        
        self._prepare_transducers()
        
        data['pml'] = self.pml.generate_grid()  # Get pml if pml is used.
        shape = self.grid.shape
        
        """Add field quantities"""
        data['field'] = dict()
        for quantity in self.FIELD_ARRAYS:
            data['field'][quantity] = np.zeros(shape, dtype=self.dtype('float'))
        
        """Add sources"""
        data['source'] = self.create_source_arrays()    # Create source terms.
        
        """Prepare receivers."""
        timesteps = self.timesteps
        for receiver in self.receivers:
            #receiver._model = self
            receiver._position_in_nodes = receiver.position_in_nodes # Store static value
            receiver.data = {quantity: np.zeros(timesteps, dtype=self.dtype('float')) for quantity in receiver.quantities}
        
        """Prepare recorder."""
        self._prepare_recorder()
        
        data['_t'] = time.clock()   # Start timing
        
        
        
        
    #@numba.autojit    
    def _post_run(self, data):
        """
        This function is performed after the simulation is finished.
        """
        logger.info('Progress: Done')
        
        if self.settings['recording']['use']:
            self._file_handler[0].close()
    
    #@numba.autojit
    def _pre_update(self, data):
        """
        This function is performed before each update.
        """
        pass
    
    #@numba.autojit
    def _post_update(self, data):
        """
        This function is performed after each update.
        """
        t = time.clock()
        dt = t - data['_t']
        data['_t'] = t
        time_left = (data['steps']-data['step']) * dt
        
        self._record(data['step'], data['field'])
        
        
        logger.info('Progress: {:3.1f}% - Step {}/{}  - Time left: {:.0f}'.format(
            float(data['step'])/float(data['steps'])*100.0, 
            data['step'], 
            data['steps'],
            time_left))
    
    def _record(self, timestep, fields):
        """
        Record quantities.
        """
        for receiver in self.receivers:
            for quantity, value in receiver.data.iteritems():
                value[timestep] = fields[quantity][receiver._position_in_nodes].real
    
        if self.settings['recording']['use']:
            f = self._file_handler[0]
            group = self._file_handler[1]
            
            x = self.settings['recording']['slice'][0]
            y = self.settings['recording']['slice'][1]

            for quantity in group['field'].keys():
                #print group[quantity][timestep].shape, fields[quantity].shape
                
                group['field'][quantity][timestep,:,:] = fields[quantity][x,y].real
    
    @abc.abstractmethod
    def _update():
        """
        Update steps to perform.
        """
        pass
    
    #@numba.autojit
    def run(self):
        """
        Run simulation.
        """
        shape = self.grid.shape
        
        data = {}
        data['steps'] = self.timesteps
        data['spacing'] = self.grid.spacing
        data['timestep'] = self.timestep #* np.ones(shape, dtype=self.dtype('float'))

        self._pre_run(data)
        
        for step in xrange(data['steps']):
            data['step'] = step
            self._pre_update(data)
            self._update(data)
            self._post_update(data)
            
        self._post_run(data)
        
        self.data = data
        #return data
    
    def _attributes(self):
        model_fields = ['f_max', 'time', 'cfl', 'timestep', 'timesteps', 'sample_frequency',
                'shape_spacetime', 'shape_spacetime_with_pml', 'shape_spacetime_without_pml']
        return {item : getattr(self, item) for item in model_fields }            
        

    def to_yaml(self, filename):
        """
        Write configuration to yaml file.
        """
        
        sources = [source._attributes() for source in self.sources]
        receivers = [receiver._attributes() for receiver in self.receivers]
        
        
        data = {'model' : self._attributes(),
                'pml' : self.pml._attributes(),
                'grid' : self.grid._attributes(),
                'sources' : sources,
                'receivers' : receivers,
                'settings' : self.settings,
            }
        
        #sources = [{item : getattr(source, item) for item in source_fields } for source in self.sources]
        #receivers = [{item : getattr(receiver, item) for item in receiver_fields } for receiver in self.receivers]
        
        #data = {'model' : model,
                #'grid' : grid,
                #'pml' : pml,
                #'sources' : sources,
                #'receivers' : receivers,
                #}
        
        
        with open(filename, 'w') as fout:
            fout.write( yaml.dump(data, default_flow_style=False) )
        
        #yaml.dump(data, filename, default_flow_style=False)
        
        
        
    
    def overview(self):
        """
        Overview of settings.
        """
        return ("Simulation time: {} \n".format(self.time) + 
                "Model timesteps: {} \n".format(self.timesteps) +
                "Model timestep: {} \n".format(self.timestep) +
                "Maximum frequency: {:.1f}\n".format(self.f_max) +
                "Grid spacing: {:.3f}\n".format(self.grid.spacing) +
                "Grid shape: {}\n".format(self.grid.shape) +
                "Grid shape without PML: {}\n".format(self.grid.shape_without_pml) +
                "Grid shape with PML: {}\n".format(self.grid.shape_with_pml) +
                "PML nodes: {}\n".format(self.pml.nodes) +
                "PML depth target: {:.2f}\n".format(self.pml.depth_target) +
                "PML depth actual: {:.2f}\n".format(self.pml.depth) +
                "Grid size: {}\n".format(self.grid.size) +
                "Grid size without PML: {}\n".format(self.grid.size_without_pml) +
                "Grid size with PML: {}\n".format(self.grid.size_with_pml)
                )
            

    def create_source_arrays(self):
        """
        Create arrays describing source terms.
        
        .. note:: For now, only source at t=0 are supported.
        
        """  
        
        timesteps = self.timesteps
        #shape = self.grid.shape
        spacing = self.grid.spacing
        dimensions = self.axes.DIMENSIONS
        ndim = self.axes.ndim
        
        c = self.medium.soundspeed_for_calculation
        
        sources = dict()
        #sources['p'] = SparseList(length=timesteps, default=0.0)
        #sources['p'] = {i : 0.0 for i in range(timesteps)}
        
        shape = self.shape_spacetime
        
        #sources['p'] = SparseArray(shape, dtype=self.dtype('float'))
        sources['p'] = SparseList(length=timesteps, default=0.0)
        
        sources['v'] = dict()
        for dim in dimensions:
            sources['v'][dim] = SparseList(length=timesteps, default=0.0) # List where each item represents the emission at a certain timestep.
            #sources['v'][dim] = SparseArray(shape, dtype=self.dtype('float'))
            
        #### Create empty sparse "matrices" for each field.
        ###p_x = sp.lil_matrix((self.axes.y.n, self.axes.x.n), dtype=self.dtype)
        ###p_y = sp.lil_matrix((self.axes.y.n, self.axes.x.n), dtype=self.dtype)
        ###v_x = sp.lil_matrix((self.axes.y.n, self.axes.x.n), dtype=self.dtype)
        ###v_y = sp.lil_matrix((self.axes.y.n, self.axes.x.n), dtype=self.dtype)
        
        # Add the source contributions to these sparse matrices.
        for source in self.sources:
            #source._model = self # HACK Should already be set when source is assigned to model.
            
            if source.quantity == 'pressure':
                #sources['p'] = sources['p'] + source.field
                for step in range(timesteps):
                    sources['p'][step] += source.field[step]
                
            elif source.quantity == 'velocity':
                #sources['v'][source.component] = sources['v'][source.component] + source.field
                for step in range(timesteps):
                    sources['v'][source.component][step] += source.field[step]

            #print sources['p']

            #if source.pulse:
                #if source.pressure:
                    #sources['p'][0] += source.pressure_field_pulse()
                    
                #if source.velocity:
                    #for dim, velocity in zip(dimensions, source.velocity_field_pulse()):
                        #sources['v'][dim][0] += velocity
            
            #else: # Continues signal
                #if source.pressure is not None:
                    #for timestep in range(timesteps):
                        #sources['p'][timestep] = sources['p'][timestep] + source.pressure_field_continuous_signal()
                
                #if source.velocity is not None:
                    #for dim, velocity in zip(dimensions, source.velocity_field_continuous_signal()):
                        #sources['v'][dim][timestep] += source.pressure_field_continuous_signal()
            
                #for i, dim in enumerate(dimensions):  # For each component/dimension
                    #sources['v'][dim][0] += source.force(source.velocity[i], c, spacing) 

        #### Convert the LIL sparse matrices to a format that is more efficient for multiplication, like CSR or CSC.
        ###for name, value in sources.iteritems():
            ###sources[name] = sp.csr_matrix(value)
        
        return sources
    
    
    def plot_scene(self, filename=None):
        """
        Plot the sources and receivers on the grids.
        """
        
        self._prepare_transducers()
        
        fig = plt.figure(figsize=(16,12), dpi=80)
        
        
        
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_title("Scene")
        
        """Draw sources"""
        x = [source.position.x for source in self.sources]
        y = [source.position.y for source in self.sources]
        ax.scatter(x, y, s=100, label='Source', marker='x', color='r')#, markersize=10)
        
        """Draw receivers"""
        x = [receiver.position.x for receiver in self.receivers]
        y = [receiver.position.y for receiver in self.receivers]
        ax.scatter(x, y, s=100, label='Receiver', marker='o', color='g')#s, markersize=10)
        
        """Draw PML."""
        ax.add_patch(plt.Rectangle((0.0, 0.0), self.axes.x.length, self.pml.depth))
        ax.add_patch(plt.Rectangle((0.0, self.axes.y.length), self.axes.x.length, -self.pml.depth))
        
        ax.add_patch(plt.Rectangle((0.0, 0.0), self.pml.depth, self.axes.y.length))
        ax.add_patch(plt.Rectangle((self.axes.x.length, 0.0), -self.pml.depth, self.axes.y.length))
        
        ax.grid()
        ax.legend(scatterpoints=1)
        ax.set_xlim(0.0, self.axes.x.length)
        ax.set_ylim(0.0, self.axes.y.length)
        
        if filename:
            fig.savefig(filename)
        else:
            return fig
    
    
    def plot_impulse_responses(self, quantity='p', yscale='linear', filename=None):
        """
        Plot the impulse responses measured at the receivers.
        
        :param quantity: Quantity to plot.
        :param yscale: Logarithmic `log` or linear `linear` scale.
        :param filename: Optionally write figure to file.
        """
        
        fig = plt.figure(figsize=(16, 12), dpi=80)
        ax1 = fig.add_subplot(111)
        
        ax1.set_title("Impulse response")
        
        t = np.arange(self.timesteps) * self.timestep
        
        for receiver in self.receivers:
            if receiver.data.has_key(quantity):
                ir = receiver.data[quantity]
                if yscale == 'log':
                    ir = 20.0 * 10.0*np.log10(np.abs(ir))
                ax1.plot(t, ir, label=receiver.position)
        ax1.grid('on')
        ax1.legend()
        ax1.set_xlabel(r"$t$ in s")

        #print quantity
        
        if yscale == 'log':
            ax1.set_ylabel(r"$20 \log |{}|$ in {}".format(quantity, QUANTITIES[quantity]))
        else:
            ax1.set_ylabel(r"${}$ in {}".format(quantity, QUANTITIES[quantity]))

        if filename:
            fig.savefig(filename)
        else:
            return fig
    
    
    def plot_frequency_responses(self, quantity='p', xscale='linear', yscale='linear', filename=None):
        """
        Plot the frequency responses.
        
        The top figure shows the magnitude response and the bottom figure the phase response.
        
        """
        
        fig = plt.figure(figsize=(16, 12), dpi=80)
        
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        for receiver in self.receivers:
            if receiver.data.has_key(quantity):
                ir = receiver.data[quantity]
                f, tf = ir2fr(ir, 1.0/self.timestep)
                
                if xscale == 'log':
                    func = 'semilogx'
                else:
                    func = 'plot'
                    
                getattr(ax1, func)(f, 10.0*np.log10(np.abs(tf)), label=receiver.position)
                getattr(ax2, func)(f, np.unwrap(np.angle(tf)), label=receiver.position)
            
        
        ax1.set_title("Magnitude response")
        ax1.grid()
        ax1.legend()
        ax1.set_xlabel(r"$f$ in Hz")
        ax1.set_ylabel(r"$|{}|$ in {}".format(quantity, QUANTITIES[quantity]))
        #ax1.set_ylabel(r"$20 \log|{}|$ in {}".format(quantity, QUANTITIES[quantity]))
        ax1.set_xlim(0.0, self.f_max)
        
        ax2.set_title("Phase response")
        ax2.grid()
        ax2.legend(loc="lower right")
        ax2.set_xlabel(r"$f$ in Hz")
        ax2.set_ylabel(r"$\angle {}$ in radians".format(quantity))
        ax2.set_xlim(0.0, self.f_max)

            
        #if xscale == 'log':
            #ax1.semilogx(t, receiver.data[quantity], label=receiver.position)
        #else:
        
            
        if filename:
            fig.savefig(filename)
        else:
            return fig
    
    def plot_field(self, filename=None):
        """
        Plot pressure and velocities fields.
        
        :param data: Data dictionary.
        :param filename: Optional filename.
        """
        data = self.data
        
        spacing = self.grid.spacing
        
        #xl = self.axes.x.length
        #yl = self.axes.y.length
        
        x = np.arange(self.axes.x.nodes+1) * spacing
        y = np.arange(self.axes.y.nodes+1) * spacing
        
        fig = plt.figure(figsize=(16, 12), dpi=80)
        grid = AxesGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.3, cbar_mode="each")
        
        ax1 = grid[0]#.add_subplot(131, aspect='equal')
        ax1.set_title("Sound pressure")
        plot1 = ax1.pcolormesh(x, y, (data['field']['p']).T.real)
        ax1.set_xlim(0.0, x[-1])
        ax1.set_ylim(0.0, y[-1])
        ax1.grid()
        grid.cbar_axes[0].colorbar(plot1)
        
        x += spacing/2.0
        y += spacing/2.0
        
        ax2 = grid[1]#fig.add_subplot(132, aspect='equal')
        ax2.set_title(r"Particle velocity in $x$-direction")
        plot2 = ax2.pcolormesh(x, y, data['field']['v_x'].T.real)
        #ax2.set_xlim(0.0 + spacing/2.0, xl + spacing/2.0)
        #ax2.set_ylim(0.0 + spacing/2.0, yl + spacing/2.0)
        ax2.grid()
        grid.cbar_axes[1].colorbar(plot2)
        
        ax3 = grid[2]#fig.add_subplot(133, aspect='equal')
        ax3.set_title(r"Particle velocity in $y$-direction")
        plot3 = ax3.pcolormesh(x, y, data['field']['v_y'].T.real)
        #ax3.set_xlim(0.0 + spacing/2.0, xl + spacing/2.0)
        #ax3.set_ylim(0.0 + spacing/2.0, yl + spacing/2.0)
        ax3.grid()
        grid.cbar_axes[2].colorbar(plot3)
        
        if filename:
            fig.savefig(filename)
        else:
            return fig
    
DEFAULT_SETTINGS = {

    'pml' : {
        'use' : True, # Whether to use a PML. The field arrays will be shaped accordingly.
        'include_in_output' : True, # Include the PML in the output values/grids.
        },
    #'multithreading': {
        #'use' : False,
        #'threads' : 2,
        #},
    'recording' : {
        'use' : False,  # Use recorder
        'filename' : 'recording.hdf',
        'groupname' : None,
        'quantities' : ['p'],
        'slice' : (slice(None), slice(None)), # Can be a (x,y) or (x,y,z) tuple.
        'meta' : True # Store addition data like speed of sound, grid shape and size, etc.
        },
    }
"""
Default settings.
"""




def decibel_to_neper(decibel):
    """
    Convert decibel to neper.
    
    :param decibel: Value in decibel (dB).
    
    The conversion is done according to
    
    .. math :: \\mathrm{dB} = \\frac{\\log{10}}{20} \\mathrm{Np}
    
    """
    return np.log(10.0) / 20.0  * decibel

def neper_to_decibel(neper):
    """
    Convert neper to decibel.
    
    :param neper: Value in neper (Np).
    
    The conversion is done according to

    .. math :: \\mathrm{Np} = \\frac{20}{\\log{10}} \\mathrm{dB}
    """
    return 20.0 / np.log(10.0) * neper

def CFL(c_0, timestep, spacing):
    """
    Courant-Friedrichs-Lewy number.
    
    :param c_0: Speed of sound :math:`c_0`.
    :param timestep: Time step :math:`\\Delta t`.
    :param spacing: Spatial resolution :math:`\\Delta x`.
    
    .. math:: \\mathrm{CFL} = c_0 \\frac{\\Delta t}{\\Delta x}
    
    """
    return c_0 * timestep / spacing

def wavenumbers(n, spacing):
    """
    Calculate ``n`` wavenumbers for a given ``spacing``.
    
    :param n: Grid points
    :param spacing: Spacing
    
    Equation 2.17e.
    """
    return np.fft.fftfreq(n, spacing) * 2.0 * np.pi

def frequencies(n, fs):
    """
    Frequency vector for FFT.
    
    See also :func:`np.fft.fftfreq`.
    
    :param n: Amount of frequencies.
    :param fs: Sample frequencies
    """
    return np.fft.fftfreq(n, fs)
    
def circular_receiver_array(center, radius, n, quantities=None):
    """
    Create a receiver array with receivers on a circle. Returns a list of receivers.
    
    .. note:: 2D only.
    
    """
    quantities = quantities if quantities else ['p']
    
    receivers = list()
    
    angles = np.linspace(0, 2.0*np.pi, n, endpoint=False)
    
    x_vec = np.cos(angles) * radius
    y_vec = np.sin(angles) * radius
    
    for x, y in zip(x_vec, y_vec):
        receivers.append(Receiver(Position2D(center.x + x, center.y + y), quantities))
    
    return receivers

def line_receiver_array(start, stop, n, quantities=None):
    """
    Create an array of receivers along a line. Returns a list of receivers.
    
    .. note:: 2D only.
    """
    quantities = quantities if quantities else ['p']
    
    receivers = list()
    
    p_x = np.linspace(start.x, stop.x, n, endpoint=True)
    p_y = np.linspace(start.y, stop.y, n, endpoint=True)
    
    for x, y in zip(p_x, p_y):
        receivers.append(Receiver(Position2D(x, y), quantities))
    
    return receivers
    

def plot_signal(signal, time, filename=None):
    """
    Plot signal.
    
    :param signal: Signal vector.
    :param time: Time vector.
    :param filename: Optional filename.
    """
    fig = plt.figure()
    ax0 = fig.add_subplot(111)
    ax0.set_title('Signal')
    ax0.plot(self.time, self.signal)
    ax0.set_xlabel(r'$t$ in s')
    ax0.set_ylabel(r'$x$ in -') 

    if filename:
        fig.savefig(filename)
    else:
        return fig
    
    
def animate_field(data, fps, filename=None):
    """
    Create animation of field.
    """
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import animation

    
    # animation function.  This is called sequentially
    def animate(i, d):
        return plt.pcolormesh(d[i])

    fig = plt.figure()

    anim = animation.FuncAnimation(fig, animate, frames=data.shape[0], fargs=[data])

    if filename:
        anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()
    
    
def ir2fr(ir, fs, N=None):
    """
    Convert impulse response into frequency response. Returns single-sided RMS spectrum.
    
    :param ir: Impulser response
    :param fs: Sample frequency
    :param N: Blocks
    
    Calculates the positive frequencies using :func:`np.fft.rfft`.
    Corrections are then applied to obtain the single-sided spectrum.
    
    .. note:: Single-sided spectrum.
    
    """
    #ir = ir - np.mean(ir) # Remove DC component.
    
    N = N if N else ir.shape[-1]
    fr = rfft(ir, n=N) / N
    f = np.fft.rfftfreq(N, 1.0/fs)    #/ 2.0
    
    fr *= 2.0
    fr[..., 0] /= 2.0    # DC component should not be doubled.
    if not N%2: # if not uneven
        fr[..., -1] /= 2.0 # And neither should fs/2 be.
    
    #f = np.arange(0, N/2+1)*(fs/N)
    
    return f, fr
    

QUANTITIES = {
    'p' : 'Pa',
    'v' : 'm/s'
    }