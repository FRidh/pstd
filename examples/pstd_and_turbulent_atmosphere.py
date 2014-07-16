"""
Example showing how to use the k-space PSTD method.
"""
import numpy as np

import matplotlib.pyplot as plt

from acoustics.pstd import PSTD as Model
from acoustics.pstd import Medium, PML, Position2D, Source
from acoustics.turbulence import VonKarman2DTempWind

import logging

f_max = 500.0

f_margin = 1.0

#def main():

"""Given parameters."""
N = 100                     # Amount of modes
k_max = 10.0                # Maxmimum wavenumber

"""Parameters for Von Karman spectrum."""
K_0 = 1.0/10.0
L = 2.0 * np.pi / K_0       # Size of largest eddy.
C_T = np.sqrt(1e-7)         # 
T_0 = 1.0                   #
c_0 = 343.2
C_v = c_0 * 0.001



"""Other settings."""
wavenumber_resolution = k_max / N
spatial_resolution = c_0 / (2.0 * f_max * f_margin)   # We don't need it for the calculations but we do need it to create an instance.

#print spatial_resolution


x = 100.0
y = 0.0
z = 100.0

vk = VonKarman2DTempWind(max_mode_order=N, 
                    L=L,
                    C_T=C_T,
                    T_0=T_0,
                    C_v=C_v,
                    c_0=c_0,
                    wavenumber_resolution=wavenumber_resolution,
                    spatial_resolution=spatial_resolution,
                    x=x,
                    y=y,
                    z=z
                    )


mu = vk.field()
#mu = 0.0

#print  mu

#print("Mu shape: {}".format(mu.shape))
c = ( mu + 1.0 ) * c_0
#c = c_0
#print c

time = np.sqrt(x*x + z*z) / np.min(c_0)

print("Time: {:.2f}".format(time))

CFL = 0.3
density = 1.296

f_max = np.max(c) / ( spatial_resolution * 2.0 )

"""To show the progress of the calculation we use the INFO log"""
logger = logging.getLogger('acoustics.pstd')    # Module name
logger.setLevel(logging.INFO)                   # Logging level

medium = Medium(c, density)   # Medium

pml = PML((10.0, 10.0), depth=1.0)               # Perfectly Matched Layer

source = Source(Position2D(x/2.0, z/2.0), pressure=0.001)

model = Model(size=[x, z], time=time, f_max=f_max, medium=medium, cfl=CFL, pml=pml)

model.spacing = spatial_resolution
model.sources.append(source)


#print("Model size: {}".format(model.grid.size)))
print("Model timesteps: {}".format(model.timesteps))
print("Model timestep: {}".format(model.timestep))
print("Grid spacing: {:.3f}".format(model.grid.spacing))
print("Maximum frequency: {:.1f}".format(model.f_max))
print("Axis x nodes: {}".format(model.axes.x.nodes))
print("Axis y nodes: {}".format(model.axes.y.nodes))
print("Grid shape: {}".format(model.grid.shape))
#print("Grid shape with PML: {}".format(model.grid.shape_with_pml))
print("PML nodes: {}".format(model.pml.nodes))
print("PML depth target: {:.2f}".format(model.pml.depth_target))
print("PML depth actual: {:.2f}".format(model.pml.depth))
print("Grid size: {}".format(model.grid.size))
#print("Grid size with PML: {}".format(model.grid.size_with_pml))


#print model.medium.soundspeed.shape
print model.axes.x.nodes_without_pml
print model.axes.x.nodes_with_pml
print model.axes.x.nodes


model.timesteps = 400

#model.settings['pml']['use'] = False

data = model.run()


fig = plt.figure(figsize=(16, 12), dpi=80, aspect='equal')
ax1 = fig.add_subplot(221)
ax1.set_title("Pressure x")
ax1.pcolor(data['field']['p_x']+data['field']['p_y'])
ax2 = fig.add_subplot(223)
ax2.set_title("Velocity x")
ax2.pcolor(data['field']['v_x'])
ax3 = fig.add_subplot(224)
ax3.set_title("Velocity y")
ax3.pcolor(data['field']['v_y'])

fig.savefig("pstd.png")


#if __name__ == '__main__':
    #main()
