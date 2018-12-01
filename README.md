# CUDA-accelerated Fast Faraday Synthesis (cuFFS)

Dependencies
============
* [gcc](https://gcc.gnu.org/)
* [libconfig](http://www.hyperrealm.com/libconfig/)
* [cfitsio](http://heasarc.gsfc.nasa.gov/fitsio/fitsio.html)
* [OpenMP](https://www.openmp.org/)
* [gnuplot](http://www.gnuplot.info/) (Optional)
* [nvcc](docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) (Need both the driver and the toolkit)
* [hdf5](https://support.hdfgroup.org/HDF5/)

Installation
============
See build.sh for sample compiler instruction.

Notes
=====
* The code assumes that the pixels values are IEEE single precision floating points (BITPIX=-32)
* The input cubes must have 3 axes (2 spatial dimensions and 1 frequency axis) with frequency axis as NAXIS1. If you have individual stokes Q and U channel maps, use the helper/makeFitsCube.py to get the data in the required format.
