## Computational Physics Homework 3
This repository contains my solutions to Homework 3 of Computational Physics. These can be found in the folder titled 'homework3'
## Contents
'homework3/'
- 'sunspots_periodicity.py': Detects periodicity in monthly sunspot numbers using a discrete Fourier transform on 'sunspots.txt'. The dominant frequency corresponds to the 11-year solar cycle.
- 'dow_filtering.py': Performs Fourier filtering and smoothing on 'dow.txt', the Dow Jones Industrial Average data. Generates smoothed signals by keeping only the lowest 10% and 2% of Fourier modes.
- 'deconvolution_2d.py': Implements 2D image deconvolution of a blurred grayscale image ('blur.txt') using Fourier transforms and a Gaussian point spread function with σ = 25 and ε = 10⁻³.
-'sunspots.txt', 'dow.txt', and 'blur.txt' are the data files provided for each problem.
- the .png files are all of the generated images and plots.
