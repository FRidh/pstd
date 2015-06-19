from setuptools import setup, find_packages

CLASSIFIERS = [
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only'
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        ]


setup(
      name='pstd',
      version='0.1',
      description="K-space PSTD implementation for Python.",
      long_description=open('README.md').read(),
      author='Frederik Rietdijk',
      author_email='fridh@fridh.nl',
      license='LICENSE',
      packages=find_packages(),
      scripts=[],
      zip_safe=False,
      classifiers=CLASSIFIERS,
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'numexpr',
          ],
      extras_require={
          'hdf5' : 'h5py',
          'jit' : 'numba',
          'Fast FFT' : 'pyFFTW',
          'YAML' : 'PyYAML',
          }
      )
