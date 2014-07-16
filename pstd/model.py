"""
This module contains an implementation of the k-space PSTD method.
"""
import yaml
import numpy as np
import abc
import collections
import time
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable
import weakref
import itertools
from acoustics import Signal


import logging
#logging = logging.getLogger(__name__)    # Use module name as logging name

#import h5py
from acoustics.signal import ir2fr

#try:
    #import numba
#except ImportError:
    #raise ImportWarning("Cannot import numba. JIT not available.")

#try:
    #from pyfftw.interfaces.numpy_fft import rfft
#except ImportError:
    #from numpy.fft import rfft

#from sparse_cython.core import SparseArray

#from multiprocessing import cpu_count

#class SparseList(dict):
    #"""
    #Sparse list.
    
    #.. note:: Quick implementation. Better approach would be to encapulse the dictionary.
    
    #"""
    
    #def __init__(self, length, default=0.0):
        
        #self._length = length
        #self._default = default
    
    #def __getitem__(self, i):
        #if i >= self._length:
            #raise IndexError("Out of range.")
        #else:
            #try:
                #return dict.__getitem__(self, i)
            #except KeyError:
                #return self._default

    #def __len__(self):
        #return self._length
    
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
    
    
class Name(object):
    """
    Unique Name descriptor.
    """
    
    #def __init__(self):
        #pass
    
    def __get__(self, instance, owner):
        try:
            return instance.__dict__['name']
        except KeyError:
            return None
        
    def __set__(self, instance, value):
        if instance.name is None:
            """Set unique name."""
            names = (obj.name for obj in instance.model.objects)
            if value in names:
                msg = 'Name {} is not unique.'.format(str(value))
                warnings.warn(msg, Warning)
                value += '1'
            instance.__dict__['name'] = value
        else:
            raise ValueError("Cannot change name.")
        

class Axes(object, metaclass=abc.ABCMeta):
    """Axes base class.
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
    """Axes for two-dimensional model.
    """
    
    DIMENSIONS = ['x', 'y']

class Axes3D(Axes):
    """Axes for three-dimensional model.
    """
    
    DIMENSIONS = ['x'], ['y'], ['z']


class Axis(object):
    """Axis of :class:`Model`.
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
        
    @property
    def wavenumbers(self):
        """
        Wavenumbers.
        """
        return wavenumbers(self.nodes, self.spacing).astype(self._axes._model.dtype('float'), copy=False)


Position2D = collections.namedtuple("Position2D", ['x', 'y'])
"""Position in 2D
"""

Position3D = collections.namedtuple("Position3D", ['x', 'y', 'z'])
"""Position in 3D
"""

class Transducer(object):
    
    #name = Name()
    #"""
    #Unique name of object.
    #"""
        
    def __init__(self, model, name, position):
        
        #super().__init__()
        
        self._model = model
        """
        Reference to model.
        """
        
        self.name = name
        
        self.position_target = position
        
    def _start(self):
        """
        Method is called when a new simulation is started to do some preparations.
        """
        
        self._position_in_nodes = self.position_in_nodes
        """
        Position in nodes. Static value that is updated during pre-run.
        """
    
    @property
    def position_target(self):
        """Position (center) of the source. See :class:`Position`.
        """
        return self._position
        
    @position_target.setter
    def position_target(self, position):
        self._position = Position2D(*position)
    
    
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
    """Mass.
    
    :param pressure: Sound pressure
    :param c: Speed of sound
    :param spacing: Spacing
    :param ndim: Amount of dimensions.
    
    .. math:: S_{M_i} = p \\frac{2 }{c N \\Delta x}
    
    where :math:`i` represents the axis.
    """
    return pressure * 2.0 / (c * ndim * spacing)


def force(velocity, c, spacing):
    """Force.
    
    :param velocity: Particle velocity.
    :param c: Speed of sound.
    :param spacing: Spacing
    
    .. math:: S_{F_i} = u_i \\frac{2 c}{\\Delta x}
    
    where :math:`i` represents the axis.
    """
    return velocity * 2.0 * c / spacing
  

class Source(Transducer):
    """Source
    """
    
    _field_generator = None
    """
    Generator that is used during the simulation. See :meth:`_start`.
    """
    
    def __init__(self, model, name, position, quantity, component=None):
        
        super().__init__(model, name, position)
        
        self.quantity = quantity
        """Quantity of source. Pressure or velocity.
        """
        
        self.component = component
        """Component of field quantity.
        """
        
    @abc.abstractmethod    
    def _field(self):
        """
        Source field generator.
        
        This generator yields every timestep a scalar or array.
        """
        yield
    
    def _start(self):
        super()._start()
        self._field_generator = self._field()
        
#class FieldSource(object):
    #"""
    #Custom source field.
    #"""
    #def __init__(self, quantity, field, component=None):
        #super(Field, self).__init__(self, quantity, field)
        #self.field = field

class PointSource(Source):
    """
    Point source.
    """
    
    def __init__(self, model, name, position=None, quantity=None, component=None, excitation='pulse', amplitude=None, frequency=None):
        
        super().__init__(model, name, position, quantity, component)

        self.excitation = excitation
        """Type of excitation. Options are `'sine'` and `'pulse'`.
        """
        
        self.amplitude = amplitude
        
        self.component = component
        """Component of field ('x', 'y').
        """
        
        self.frequency = frequency
        """Frequency of signal in case of a sine excitation.
        """

        
    def _amplitude(self):
        if self.quantity=='pressure':
            return self.mass
        elif self.quantity=='velocity':
            return self.force
        else:
            raise ValueError("Incorrect quantity.")
        

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
        else:
            raise ValueError("This is not a pressure/mass source.")
        
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
        else:
            raise ValueError("This is not a velocity/force source.")
        
    def _attributes(self):

        source_fields = ['position', 'position_with_pml', 'position_without_pml', 
                         'position_in_nodes', 'position_in_nodes_with_pml', 'position_in_nodes_without_pml',
                         'quantity', 'component', 'excitation', 'amplitude', 
                         'frequency', 'mass', 'force']        
    
        return { item : getattr(self, item) for item in source_fields}

    def _field(self):
        """
        Field.
        """
        if self.excitation=='sine':
            yield from self._field_sine()
        elif self.excitation=='pulse':
            yield from self._field_pulse()
        #elif self.excitation=='custom':
            #yield from self._field_custom()
        else:
            raise ValueError("Excitation type is not specified.")

    def _field_sine(self):
        """
        Sinusoidal signal.
        
        This is a generator returning a sinusoidal with frequency :attr:`frequency`.
        
        """
        position = self.position_in_nodes
        shape = self._model.grid.shape
        amplitude = self._amplitude()
        timestep = self._model.timestep
        frequency = self.frequency
        dtype = self._model.dtype('float')
        
        i = 0
        while True:
            time = i * timestep
            field = np.zeros(shape, dtype=dtype)
            field[(position[0], position[1])] = amplitude * np.sin(2.0*np.pi*frequency * time)
            yield field
            i+=1
        
    def _field_pulse(self):
        """
        Gaussian pulse at :math:`t=0`.
        
        This is a generator returning a pulse at :math:`t=0` and 0 for all other times.
        
        """
        position = self.position_in_nodes
        amplitude = self._amplitude()
        spacing = self._model.grid.spacing      # Grid spacing
        
        # We now calculate a small pulse, and then fit it in
        pulse = self.gaussian_pulse(amplitude, spacing)  # Small pulse
        shape = pulse.shape
        offset = tuple(int(round(i/2.0)) for i in shape)
        
        pulse_grid = np.zeros(self._model.grid.shape, dtype=self._model.dtype('float'))   # Grid
        try:
            pulse_grid[ position[0]-offset[0]: position[0]-offset[0]+shape[0], position[1]-offset[1] : position[1]-offset[1]+shape[1] ] = pulse
        except ValueError:
            raise ValueError("Source is too close to the border. Cannot fit Gaussian pulse.")
        
        yield pulse_grid
        while True:
            yield 0.0

class Receiver(Transducer):
    """
    Receiver.
    """

    def __init__(self, model, name, position, quantity='pressure', component=None, last_value_only=False, filename=None):
        
        super().__init__(model, name, position)
        
        self.last_value_only = last_value_only
        """
        Record only the final value (True) or the impulse response (False).
        """
        
        self.quantity = quantity
        """Quantity to record.
        """
        
        self.component = component
        """Field component to record.
        """
        
        self._data = list()
        """Store received values. This value will be reset 
        whenever the simulation is restarted.
        """
        
        #self.quantities = quantities if quantities else ['pressure']
        """
        List of quantities to record.
        """
        
        
        #self.filename = filename
        #"""
        #Store in file.
        #"""
        
        #self._data = dict()
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


    def _start(self):
        super()._start()
        #self.data = {quantity: list() for quantity in self.quantities}
        self._data = list()
    
    def _attributes(self):
        receiver_fields = ['position', 'position_with_pml', 'position_without_pml', 
                         'position_in_nodes', 'position_in_nodes_with_pml', 'position_in_nodes_without_pml',
                         'quantities', 'last_value_only']
        
        return { item : getattr(self, item) for item in receiver_fields}
    
    
    def recording(self):
        """Get recording.
        """
        return Signal(self._data, fs=self._model.temporal_sample_frequency)
            
    
class ReceiverArray(object):
    """
    Receiver array.
    """
    
    def __init__(self, quantities=None, last_value_only=False, filename=None):
        
        self.quantities = quantities if quantities else ['pressure']
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
    
    
objects_map = {"PointSource" : PointSource,
               "Receiver"    : Receiver,
               }    
"""
Map with transducers.
"""
    

class Medium(object):
    """
    Medium. 
    
    See also :class:`acoustics.atmosphere.Atmosphere`.
    """
    
    def __init__(self, soundspeed=343.0, density=1.296):#, refractive_index=1.0):
        
        self.soundspeed = soundspeed
        """
        Speed of sound :math:`c_0`.
        
        .. note:: In case of a inhomogeneous atmosphere, this value is an array.
        
        """
        self.density = density
        """
        Density :math:`\\rho`.
        """    
    
        self._model =  None
    
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
        
        if self._model is None:
            raise ValueError("Cannot return value. Medium needs to be part of a Model.")
        
        elif len(self.soundspeed) == 1 or self.soundspeed.shape == self._model.grid.shape: # Generally the case we have no PML.
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
    
    def __init__(self, absorption_coefficient=None, depth=0.0):
        
        self.absorption_coefficient = absorption_coefficient if absorption_coefficient else (0.0, 0.0)
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
        
        .. note:: 2D only!
        
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
    
    def plot(self):
        """Plot PML.
        """
        
        pml = self.generate_grid()
        
        x = np.arange(self._model.axes.x.nodes+1) * self._model.grid.spacing
        y = np.arange(self._model.axes.y.nodes+1) * self._model.grid.spacing
        xl = self._model.axes.x.length
        yl = self._model.axes.y.length
        
        fig = plt.figure()
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
            return np.min(self._model.medium.soundspeed) / self.spatial_sample_frequency #(2.0 * self._model.maximum_frequency)
            
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
        return 2.0 * self._model.maximum_frequency
    
    def _attributes(self):
        grid_fields = ['size', 'size_with_pml', 'size_without_pml', 'shape',
                'shape_with_pml', 'shape_without_pml', 'spacing', 
                'spatial_sample_frequency']

        return {item : getattr(self, item) for item in grid_fields }
        
        
class Model(object, metaclass=abc.ABCMeta):
    """
    Abstract model for Time-Domain simulations.
    """
    
    precision = 'double'
    """
    Floating point precision. Valid values are 'single' and 'double' for respectively single and double precision.
    """
    
    #_file_handler = None
    #"""
    #HDF5 file handler.
    #"""
    
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

    #axes = None
    #"""
    #Axes in the model. See :class:`Axes` and :class:`Axis`.
    #"""
        
    grid = ReverseReference(attr='grid', remote='_model')
    """Grid. See :class:`Grid`.
    """
    pml = ReverseReference(attr='pml', remote='_model')
    """Perfectly Matched Layer. See :class:`PML`.
    """
    axes = ReverseReference(attr='axes', remote='_model')
    """Axes. See :class:`Axes`.
    """
    medium = ReverseReference(attr='medium', remote='_model')
    """Medium. See :class:`Medium`.
    """
    
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
    
    def __init__(self, maximum_frequency, medium=None, pml=None, cfl=0.05, spacing=None, axes=None, size=None, settings=None, grid=None):
        
        
        super(Model, self).__init__()
        
        self.settings = dict()
        """Settings.
        
        .. seealso:: :attr:`DEFAULT_SETTINGS`
        
        """
        self.settings.update(DEFAULT_SETTINGS)
        
        if settings:
            self.settings.update(settings) # Apply given settings.
        
        
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
        """Grid. 
        
        .. seealso:: :class:`Grid`.
        
        """

        self.maximum_frequency = maximum_frequency
        """Upper frequency limit of the model. Above this frequency limit the model is not reliable.
        """
        
        self.medium = medium if medium else Medium()
        """Medium. 
        
        .. seealso:: :class:`Medium`.
        
        """
        
        self.cfl = cfl
        
        self.pml = pml if pml else PML()
        """Perfectly Matched Layer.
        
        .. seealso:: :class:`PML`
        
        """

        
        self._objects = list()
        """Private list of transducers in model.
        """

    @property
    def objects(self):
        """Object.
        """
        yield from (self.getObject(obj.name) for obj in self._objects)

    @property
    def sources(self):
        """Sources. See :class:`Source`.
        """
        yield from (obj for obj in self.objects if isinstance(obj, Source))
        
    @property
    def receivers(self):
        """Receivers. See :class:`Receiver`.
        """
        yield from (obj for obj in self.objects if isinstance(obj, Receiver))

    def getObject(self, name):
        """Get object by name.
        
        :param name: Name of `object`.
        
        :returns: Proxy to `object`.
        
        """
        name = name if isinstance(name, str) else name.name
        
        for obj in self._objects:
            if name == obj.name:
                return weakref.proxy(obj) 
        else:
            raise ValueError("Unknown name.")
    
    def addObject(self, name, sort, position, **kwargs):
        """Add object to model.
        """
        obj = objects_map[sort](weakref.proxy(self), name, position, **kwargs)
        self._objects.append(obj)
        return self.getObject(obj.name)
    
    #def addSource(self, name):
        #pass
    
    #def addReceiver(self, name):
        #pass
    
    def removeObject(self, name):
        """Delete object from model.
        
        :param name: Name of object.
        
        """
        for obj in self.objects:
            if name == obj.name:
                self._objects.remove(obj)
    
    @property
    def cfl(self):
        """Courant-Friedrichs-Lewy number. The CFL number can be calculated using :func:`cfl`.
        """
        return self._cfl

    @cfl.setter
    def cfl(self, x):
        if not x <= np.mean(self.medium.soundspeed)/np.max(self.medium.soundspeed):
            raise ValueError("CFL too high.")
        self._cfl = x
        
    @property
    def constant_field(self):
        """Constant refractive-index field.
        """
        try:
            if self.medium.soundspeed.var() != 0.0:
                return False
            else:
                return True
        except AttributeError:
            return True
    
    @property
    def ndim(self):
        """Amount of dimensions. This value is determined from the amount of axes.
        """
        return len(self.axes)

    _timestep = None

    @property
    def timestep(self):
        """Timestep.
        
        .. math:: \\Delta t = \\frac{ \\mathrm{CFL} \\Delta x } {c_{max}}
        
        """
        if self._timestep:
            return self._timestep
        else:
            return self.cfl * self.grid.spacing / float(np.max(self.medium.soundspeed))

    @property
    def temporal_sample_frequency(self):
        """Temporal sample frequency in Hz. Inverse of :attr:`timestep`.
        """
        return 1.0 / self.timestep

    #def _prepare_recorder(self):
        #"""Prepare recorder.
        #"""
        
        #if self.settings['recording']['use']:
            #f = h5py.File(self.settings['recording']['filename'],'a')
            
            #if self.settings['recording']['groupname']:
                #groupname = self.settings['recording']['groupname']
            #else:
                #groupname = datetime.datetime.utcnow().replace(microsecond=0).isoformat()
            
            #group = f.create_group(groupname)
            
            #self._file_handler = (f, group)
            
            ##shape = self.shape_spacetime
            
            #shape = list()
            #for d, length in zip(self.settings['recording']['slice'], self.grid.shape):
                #if isinstance(d, int):
                    #shape.append(1)
                #elif isinstance(d, slice):
                    #shape.append(_slice_length(d, length))
                    
            ##shape = [i for i in self.settings['recording']['slice']]
            #shape.insert(0, self.timesteps)
            
            #shape = tuple(shape)
            ##shape = (self.timesteps, 
                     ##self.settings['recording']['slice'][0],
                     ##self.settings['recording']['slice'][1])

            #field = group.create_group('field')
            #for quantity in self.settings['recording']['quantities']:
                #field.create_dataset(quantity, shape, dtype=self.dtype('float'))

            #if self.settings['recording']['meta']:
                ## Main parameters
                
                #model = group.create_group('model')
                #for key, value in self._attributes().items():
                    #model.attrs[key] = value
                    
                ##group.attrs.create('cfl', self.cfl)
                ##group.attrs.create('ndim', self.ndim)
                ##group.attrs.create('timestep', self.timestep)
                ##group.attrs.create('timesteps', self.timesteps)
                ##group.attrs.create('time', self.time)
                ##group.attrs.create('temporal_sample_frequency', self.sample_frequency)
                
                ## Grid parameters
                #grid = group.create_group('grid')
                #for key, value in self.grid._attributes().items():
                    #grid.attrs[key] = value
                
                ## PML parameters
                #pml = group.create_group('pml')
                #for key, value in self.pml._attributes().items():
                    #pml.attrs[key] = value
                
                
                ##grid.attrs.create('spacing', self.grid.spacing)
                ##grid.attrs.create('spatial_sample_frequency', self.grid.spatial_sample_frequency)
                ##grid.attrs.create('size', self.grid.size)
                ##grid.attrs.create('size_with_pml', self.grid.size_with_pml)
                ##grid.attrs.create('size_without_pml', self.grid.size_without_pml)
                ##grid.attrs.create('shape', self.grid.shape)
                ##grid.attrs.create('shape_with_pml', self.grid.shape_with_pml)
                ##grid.attrs.create('shape_without_pml', self.grid.shape_without_pml)
                
                ## Medium parameters
                #medium = group.create_group('medium')
                #medium.create_dataset('soundspeed', data=self.medium.soundspeed)
                #medium.attrs.create('density', self.medium.density)
                #medium.attrs.create('soundspeed_mean', self.medium.soundspeed_mean)
                #medium.create_dataset('soundspeed_for_calculation', data=self.medium.soundspeed_for_calculation)
                #s = self.settings['recording']['slice']
                
                #try:
                    #c = group.create_dataset('c', data=self.medium.soundspeed[s])
                #except IndexError:
                    #c = group.create_dataset('c', data=self.medium.soundspeed)

                # PML parameters
                #pml = group.create_group('pml')
                #pml.attrs.create('is_used', self.pml.is_used)
                #pml.attrs.create(
               
    
    def _pre_start(self, data):
        """This method is run at the beginning of a simulation.
        """
        return data
    
    def _pre_run(self, data):
        """This method is run before every simulation run.
        """
        logging.info("Progress: Continuing simulation")
        
        data['_t'] = time.perf_counter()   # Start timing
        
        return data
        
    def _post_run(self, data):
        """This method is run after every simulation run.
        """
        logging.info('Progress: Done')
        return data
        #if self.settings['recording']['use']:
            #self._file_handler[0].close()
            
    def _pre_update(self, data):
        """This method is run before each update.
        """
        return data
        
    
    def _post_update(self, data):
        """This method is run after each update.
        """
        t = time.perf_counter()
        dt = t - data['_t']
        data['_t'] = t
        
        for receiver in self.receivers:
            receiver._data.append(data['field'][(receiver.quantity, receiver.component)][receiver._position_in_nodes].real)
            #for field, value in receiver.data.items():
                #value.append( data['field'][field][receiver._position_in_nodes].real)

        logging.info("Step {} is done and took {} seconds.".format(data['step'], dt))
        return data
    
    #def _record(self, timestep, fields_with_pml):
        #"""
        #Record quantities.
        #"""
        
        ## If a PML is used and if we don't want to include the PML nodes then we need to select only the non-PML part.
        #if self.pml.is_used and not self.settings['pml']['include_in_output']:
            #pml = self.pml.depth
            #shape_without_pml = tuple([slice(pml, length-pml) for length in self.grid.shape_with_pml])
            #fields = dict()
        
            #for key, value in fields_with_pml.items():
                #fields['key'] = value[shape_without_pml]
        #else:
            #fields = fields_with_pml
        
        #for receiver in self.receivers:
            #for quantity, value in receiver.data.items():
                #value[timestep] = fields[quantity][receiver._position_in_nodes].real
    
        #if self.settings['recording']['use']:
            #f = self._file_handler[0]
            #group = self._file_handler[1]
            
            #x = self.settings['recording']['slice'][0]
            #y = self.settings['recording']['slice'][1]

            #for quantity in group['field'].keys():
                ##print group[quantity][timestep].shape, fields[quantity].shape
                
                #group['field'][quantity][timestep,:,:] = fields[quantity][x,y].real
    
    @abc.abstractmethod
    def _update():
        """Update steps to perform. This needs to be implemented.
        """
        pass
    
    def run(self, steps=None, seconds=None):
        """Run the simulation for a specified amount of steps or time.
        
        :param steps: Amount of steps.
        :param seconds: Amount of time in seconds. Steps will be determined using :meth:`timestep`.
        
        """
        if steps is None and seconds is None:
            raise ValueError("Amount of steps or seconds needs to be specified.")
        elif steps is not None and seconds is not None:
            raise ValueError("Either amount of steps or seconds needs to be specified, not both.")
        elif seconds is not None:
            steps = int(np.ceil(seconds / self.timestep))
        
        logging.info("Will run for {} steps.".format(steps))
        
        try:
            self._generator
        except AttributeError:
            self.restart()
        g = self._generator
        
        for i in range(steps):
            data = next(g)
        else:
            self.data = self._post_run(data)
        
        return self
    
    def _run(self, data, sources):
        """
        Event loop.
        
        :param data: Generator containing `data` for every time instance.
        
        """
        data = self._pre_run(data)
        while True:
            data['source'] = next(sources)
            data = self._pre_update(data)
            data = self._update(data)
            data = self._post_update(data)

            yield data
            data['step'] += 1
        
    def restart(self):
        """
        Restart simulation.
        """ 
        self._start()
        return self

    def _start(self):
        """
        Tasks to perform on a new start.
        """
        
        logging.info("Progress: Starting new simulation")
        
        data = dict()
        data['spacing'] = self.grid.spacing
        data['timestep'] = self.timestep
        data['step'] = 0
        
        #self._prepare_transducers()
        
        data['pml'] = self.pml.generate_grid()  # Get pml if pml is used.
        shape = self.grid.shape
        
        """Add field quantities"""
        data['field'] = dict()
        for field in self.FIELD_ARRAYS:
            data['field'][field] = np.zeros(shape, dtype=self.dtype('float'))
            
        """Prepare transducers."""
        for obj in self.objects:
            obj._start()
        
        """Prepare recorder."""
        #self._prepare_recorder()

        """Generate the main loop."""
        sources = self._source_arrays_generator()
        #data['_t'] = time.perf_counter()
        
        data = self._pre_start(data)
        self.data = data # To allow model.run(steps=0)
        self._generator = self._run(data, sources)

        return self
    
    def _attributes(self):
        model_fields = ['maximum_frequency', 'time', 'cfl', 'timestep', 'timesteps', 'sample_frequency']
                #'shape_spacetime', 'shape_spacetime_with_pml', 'shape_spacetime_without_pml']
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
        
        with open(filename, 'w') as fout:
            fout.write( yaml.dump(data, default_flow_style=False) )
 
    def overview(self):
        """
        Overview of settings.
        """
        return ("Model timestep: {} \n".format(self.timestep) +
                "Maximum frequency: {:.1f}\n".format(self.maximum_frequency) +
                "Sample frequency temporal: {:.1f}\n".format(self.temporal_sample_frequency) + 
                "Sample frequency spatial: {:.1f}\n".format(self.grid.spatial_sample_frequency) + 
                "Grid spacing: {:.3f}\n".format(self.grid.spacing) +
                "Grid shape: {}\n".format(self.grid.shape) +
                "Grid shape without PML: {}\n".format(self.grid.shape_without_pml) +
                "Grid shape with PML: {}\n".format(self.grid.shape_with_pml) +
                "PML nodes: {}\n".format(self.pml.nodes) +
                "PML depth target: {:.2f}\n".format(self.pml.depth_target) +
                "PML depth actual: {:.2f}\n".format(self.pml.depth) +
                "Grid size: {}\n".format(self.grid.size) +
                "Grid size without PML: {}\n".format(self.grid.size_without_pml) +
                "Grid size with PML: {}\n".format(self.grid.size_with_pml) +
                "Amount of sources: {}\n".format(len(list(self.sources))) +
                "Amount of receivers: {}\n".format(len(list(self.receivers)))
                )
 
    
    def _source_arrays_generator(self):
        """
        Source arrays generator.
        
        This generator yields every timestep a dictionary containing source arrays.
        """
        while True:
            sources = dict()
            for field in self.FIELD_ARRAYS:
                sources[field] = 0.0
            for source in self.sources:
                sources[(source.quantity, source.component)] += next(source._field_generator)
                #if source.quantity == 'pressure':
                    #sources['pressure'] += next(source._field_generator)
                #elif source.quantity == 'velocity':
                    #sources['velocity'][source.component] += next(source._field_generator)
            yield sources
        

    def plot_scene(self):
        """
        Plot the sources and receivers on the grids.
        """
        
        #self._prepare_transducers()
        
        fig = plt.figure()
        
        
        
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
        
        return fig
    
    def plot_field(model, quantity='pressure', component=None):
        """
        Plot field.
        
        :param quantity: Field quantity.
        :param component: Field component.
        
        """
        
        TITLES = {
            ('pressure', None)  : 'Sound pressure',
            ('velocity', 'x')   : 'Particle velocity in $x$-direction.',
            ('velocity', 'y')   : 'Particle velocity in $y$-direction.',
            }
        
        QUANTITIES = {
            ('pressure', None)  : '$p$ in Pa',
            ('velocity', 'x')   : '$v_x$ in m/s',
            ('velocity', 'y')   : '$v_y$ in m/s',
            }
        
        data = model.data
        spacing = model.grid.spacing
        
        field = data['field'][(quantity, component)]
        
        if model.settings['pml']['use'] and not model.settings['pml']['include_in_output']:
            xs = model.axes.x.nodes_without_pml
            ys = model.axes.y.nodes_without_pml
            depth = model.pml.nodes
            field = field[+depth:-depth,+depth:-depth]
        
        x = np.arange(xs+1) * spacing
        y = np.arange(ys+1) * spacing
        
        if quantity is not 'pressure':
            x += spacing/2.0
            y += spacing/2.0
            
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')
        ax1.set_title(TITLES[(quantity, component)])
        plot1 = ax1.pcolormesh(x, y, field.real.T)
        ax1.set_xlabel(r'$x$ in m')
        ax1.set_ylabel(r'$y$ in m')
        ax1.set_xlim(0.0, x[-1])
        ax1.set_ylim(0.0, y[-1])
        ax1.grid()
        orientation = 'horizontal' if xs > ys else 'vertical'
        c = fig.colorbar(plot1, orientation=orientation, pad=0.06)
        c.set_label(QUANTITIES[(quantity, component)])
        
        return fig
    
    
    def plot_fields(self):
        """
        Plot pressure and velocities fields.
        
        """
        data = self.data
        
        spacing = self.grid.spacing
        
        x = np.arange(self.axes.x.nodes+1) * spacing
        y = np.arange(self.axes.y.nodes+1) * spacing
        
        fig = plt.figure()
        grid = AxesGrid(fig, 111, nrows_ncols=(1,3), axes_pad=0.3, cbar_mode="each")
                
        ax1 = grid[0]#.add_subplot(131, aspect='equal')
        ax1.set_title("Sound pressure")
        plot1 = ax1.pcolormesh(x, y, (data['field'][('pressure', None)]).T.real)
        ax1.set_xlim(0.0, x[-1])
        ax1.set_ylim(0.0, y[-1])
        ax1.grid()
        grid.cbar_axes[0].colorbar(plot1)
        
        x += spacing/2.0
        y += spacing/2.0
        
        ax2 = grid[1]#fig.add_subplot(132, aspect='equal')
        ax2.set_title(r"Particle velocity in $x$-direction")
        plot2 = ax2.pcolormesh(x, y, data['field'][('velocity','x')].T.real)
        #ax2.set_xlim(0.0 + spacing/2.0, xl + spacing/2.0)
        #ax2.set_ylim(0.0 + spacing/2.0, yl + spacing/2.0)
        ax2.grid()
        grid.cbar_axes[1].colorbar(plot2)
        
        ax3 = grid[2]#fig.add_subplot(133, aspect='equal')
        ax3.set_title(r"Particle velocity in $y$-direction")
        plot3 = ax3.pcolormesh(x, y, data['field'][('velocity','y')].T.real)
        #ax3.set_xlim(0.0 + spacing/2.0, xl + spacing/2.0)
        #ax3.set_ylim(0.0 + spacing/2.0, yl + spacing/2.0)
        ax3.grid()
        grid.cbar_axes[2].colorbar(plot3)
        
        return fig

#class Field(object):
    #"""
    #Field describes a field of values.
    #"""
    
    
    #def __init__(self, data):
        #pass
    
    
    #@abc.abstractmethod
    #def plot(self, quantity='pressure', filename=None):
        #pass
        

#class Field2D(Field):
    #"""
    #Field2D describes a 2D field of values.
    #"""
    #pass

    
DEFAULT_SETTINGS = {

    'pml' : {
        'use' : True, # Whether to use a PML. The field arrays will be shaped accordingly.
        'include_in_output' : False, # Include the PML in the output values/grids.
        },
    #'multithreading': {
        #'use' : False,
        #'threads' : 2,
        #},
    'recording' : {
        'use' : False,  # Use recorder
        'filename' : 'recording.hdf',
        'groupname' : None,
        'quantities' : ['pressure'],
        'slice' : (slice(None), slice(None)), # Can be a (x,y) or (x,y,z) tuple.
        'meta' : True # Store addition data like speed of sound, grid shape and size, etc.
        },
    }
"""
Default settings.
"""

def cfl(c_0, timestep, spacing):
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
    
        
    
def circular_receiver_array(model, name, center, radius, n, quantities=None):
    """
    Create a receiver array with receivers on a circle. Returns a list of receivers.
    
    :param model: Model.
    :param name: Base name of receivers. An integer is added to the base name.
    :param center: Center.
    :param radius: Radius of the circle.
    :param n: Amount of receivers.
    :param spacing: Spacing between receivers.
    :param quantities: List of quantities to record.
    
    
    .. note:: 2D only.
    
    """
    quantities = quantities if quantities else ['pressure']
    
    receivers = list()
    
    angles = np.linspace(0, 2.0*np.pi, n, endpoint=False)
    
    cx = np.cos(angles) * radius + center.x
    cy = np.sin(angles) * radius + center.y
    
    for i, (px, py) in enumerate(zip(cx, cy)):
        receivers.append(model.addObject(name+'_'+str(i), 'Receiver', Position2D(px, py), quantities=quantities))
    
    return receivers

def line_receiver_array(model, name, start, stop, n=None, spacing=None, quantities=None):
    """Create an array of receivers along a line. Returns a list of receivers.
    
    :param model: Model
    :param start: Start Position.
    :param stop: Stop position.
    :param n: Amount of receivers.
    :param spacing: Spacing between receivers.
    :param name: Base name of receivers. An integer is added to the base name.
    :param quantities: List of quantities to record.
    
    .. note:: 2D only.
    
    """
    quantities = quantities if quantities else ['pressure']
    
    if spacing and n:
        raise ValueError("Either spacing or n should be given. Not both.")
    elif spacing:
        n = int(round( (stop - start) / spacing ))
    else:
        pass
    
    receivers = list()
    
    cx = np.linspace(start.x, stop.x, n, endpoint=True)
    cy = np.linspace(start.y, stop.y, n, endpoint=True)
    
    for i, (px, py) in enumerate(zip(cx, cy)):
        receivers.append(model.addObject(name+'_'+str(i), 'Receiver', Position2D(px, py), quantities=quantities))
    return receivers
    

def grid_receiver_array(model, name, center, x, y, n=None, spacing=None, quantities=None):
    """Create an array of receivers. Returns a list of receivers.
    
    :param model: Model
    :param start: Start Position.
    :param stop: Stop position.
    :param n: Amount of receivers.
    :param spacing: Spacing between receivers.
    :param name: Base name of receivers. An integer is added to the base name.
    :param quantities: List of quantities to record.
    
    .. note:: 2D only.
    
    """
    quantities = quantities if quantities else ['pressure']
    
    if spacing and n:
        raise ValueError("Either spacing or n should be given. Not both.")
    elif n:
        spacing = np.sqrt(x*y/n)
    else:
        pass
        
    receivers = list()
    
    cx = np.arange(center.x-x/2.0, center.x+x/2.0, spacing) + spacing/2.0
    cy = np.arange(center.x-x/2.0, center.x+x/2.0, spacing) + spacing/2.0

    for i, (px, py) in enumerate(itertools.product(cx, cy)):
        receivers.append(model.addObject(name+'_'+str(i), 'Receiver', Position2D(px, py), quantities=quantities))
    return receivers
    
    
def radial_grid_receiver_array(model, name, center, radius, n, angles, quantities=None):
    """Create a radial grid array of receivers. 
    
    The radial grid is build using line arrays. 
    
    :param model: Model.
    :param name: Base name of receivers. An integer is added to the base name.
    :param center: Center.
    :param radius: Radius of the circle.
    :param n: Amount of receivers per angle.
    :param angles: Amount of angles/line receivers.
    :param spacing: Spacing between receivers.
    :param quantities: List of quantities to record.
    
    """
    
    quantities = quantities if quantities else ['pressure']
    
    receivers = list()
    
    angles = np.linspace(0, 2.0*np.pi, n, endpoint=False)
    
    cx = np.cos(angles) * radius + center.x
    cy = np.sin(angles) * radius + center.y
    
    for i, (px, py) in enumerate(zip(cx, cy)):
        outside = Position2D(px, px)
        receivers.append( line_receiver_array(model, name+"_rad_{}".format(i), center, outside, n=n, quantities=quantities))
    return receivers
   
    
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
        return fig
        #plt.show()
    

QUANTITIES = {
    'pressure' : 'Pa',
    'velocity' : 'm/s'
    }
