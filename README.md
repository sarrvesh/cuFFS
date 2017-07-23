# RMSynth_GPU

Code Status
============
* Currently being tested using NVidia Tesla P100-SXM2-16GB on the [Bracewell](https://confluence.csiro.au/display/SC/CSIRO+Accelerator+Cluster+-+Bracewell) cluster.

Dependencies
============
* [gcc](https://gcc.gnu.org/)
* [libconfig](http://www.hyperrealm.com/libconfig/)
* [cfitsio](http://heasarc.gsfc.nasa.gov/fitsio/fitsio.html)
* [gnuplot](http://www.gnuplot.info/) (Optional)
* [nvcc](docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) (Need both the driver and the toolkit)

Installation
============
See build_brinkley.sh for sample compiler instruction.

Notes
=====
* The code assumes that the pixels values are IEEE single precision floating points (BITPIX=-32)
* The input cubes must have 3 axes (2 spatial dimensions and 1 frequency axis) with frequency axis as NAXIS1. If you have individual stokes Q and U channel maps, use the helper/makeFitsCube.py to get the data in the required format.
