# Computational Physics Homework 1

This repository contains my solutions to Homework 1 of Computational Physics.

## Contents

- `code/`
  - `NumericalDerivativePS1.py`: Computes numerical derivatives of `cos(x)` and `exp(x)` using forward, central, and Richardson methods.
  - `NumericalIntegralsPS1.py`: Implements midpoint, trapezoid, and Simpson’s rule to evaluate integrals and compare errors.
  - `PowerSpectrumPS1.py`: Computes the two-point correlation function ξ(r) from the tabulated matter power spectrum P(k) and identifies the BAO peak.

- `data/`
  - `lcdm_z0.matter_pk`: Input data file containing tabulated P(k).

- `figures/`
  - Error plots for derivative and integration problems.
  - BAO correlation function plot.

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/noahringrose/Computational-Physics.git
   cd Computational-Physics/code

