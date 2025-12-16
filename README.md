# AE 370 Group Project 2 – 1D Wave Equation

This repository contains the numerical implementation and verification
studies for the AE 370 Group Project 2.

## Problem Description
We solve the one-dimensional wave equation
$\u_tt = c^2 u_xx$ on x ∈ [0, L] with homogeneous Dirichlet boundary conditions:
u(0,t) = u(L,t) = 0.

The initial conditions are:
u(x,0) = sin(πx/L),  u_t(x,0) = 0.

An analytical solution is available and is used for verification.

## Code
The file `wave_project.py` contains:
- The explicit second-order wave equation solver
- A spatial convergence study
- A CFL stability study

## How to Run
From a Python environment with NumPy and Matplotlib installed:

```bash
python wave_project.py
