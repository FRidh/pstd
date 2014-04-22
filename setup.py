from setuptools import setup, find_packages

setup(
      name='pstd',
      version='0.0',
      description="K-space PSTD implementation for Python.",
      long_description=open('README.md').read(),
      author='Frederik Rietdijk',
      author_email='fridh@fridh.nl',
      license='LICENSE',
      packages=find_packages(),
      scripts=[],
      zip_safe=False,
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'six',
          'sparse',
          ],
      extras_require=[
          'hdf5' : 'h5py',
          'jit' : 'numba',
          'Fast FFT' : 'pyfftw',
          'YAML' : 'PyYAML',
          'scipy' : 
      )
