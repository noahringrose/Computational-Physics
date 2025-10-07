# overrelax_experiment.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# Problem setup
c = 2.0
f = lambda x: 1.0 - np.exp(-c * x)
tol = 1e-6
maxiter = 10000

# simple iterative convergence test
def fixed_point_simple(x0, tol=1e-6, maxiter=10000):
    x = x0
    seq = [x]
    for n in range(maxiter):
        x_new = f(x)
        seq.append(x_new)
        if abs(x_new - x) < tol:
            return x_new, n+1, np.array(seq)
        x = x_new
    return x, maxiter, np.array(seq)

#over-relaxed iterative convergence test
def fixed_point_overrelax(x0, omega, tol=1e-6, maxiter=10000):
    x = x0
    seq = [x]
    for n in range(maxiter):
        dx = f(x) - x
        x_new = x + (1.0 + omega) * dx
        seq.append(x_new)
        if abs(x_new - x) < tol:
            return x_new, n+1, np.array(seq)
        x = x_new
    return x, maxiter, np.array(seq)

# grid of omegas to try
omegas = np.concatenate((np.linspace(-0.6, 0.9, 30), np.array([0.0, 0.3, 0.5, 0.8])))
omegas = np.unique(np.round(omegas, 6))

results = []
for omega in omegas:
    x_final, iters, seq = fixed_point_overrelax(0.5, omega, tol=tol, maxiter=maxiter)
    results.append({'omega': float(omega), 'iters': int(iters), 'converged': iters < maxiter, 'x_final': float(x_final)})

# simple case
x_simple, iters_simple, seq_simple = fixed_point_simple(0.5, tol=tol, maxiter=maxiter)

# save results and plots
outdir = Path('./overrelax_plots')
outdir.mkdir(parents=True, exist_ok=True)

# iterations vs omega
omegas_plot = np.array([r['omega'] for r in results])
iters_plot = np.array([r['iters'] for r in results])
converged_plot = np.array([r['converged'] for r in results])

plt.figure(figsize=(6,4))
mask_conv = converged_plot
plt.plot(omegas_plot[mask_conv], iters_plot[mask_conv], marker='o', linestyle='-')
plt.xlabel(r'$\omega$')
plt.ylabel('Iterations to converge (tol = 1e-6)')
plt.title('Iterations to Converge vs Overrelaxation Parameter ω\n(f(x)=1-e^{-2x}, x0=0.5)')
plt.grid(True)
plt.tight_layout()
plt.savefig(outdir / 'iters_vs_omega.png')
plt.close()

