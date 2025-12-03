import numpy as np
import matplotlib.pyplot as plt
import time
from math import sin, pi

# Parameters / data
data = np.loadtxt("/Users/noahringrose/Desktop/ComputationalPhysics/homework 5/particles.dat")
x, y = data[:,0], data[:,1]

M = 100                # grid has M x M cells
L = 100.0
dx = L / M             # cell size (should be 1.0 here)
h2 = dx*dx

epsilon0 = 8.8541878128e-12      # units as in your code
target_tol = 1e-10

# solver limits
max_iter_relax = 200000
max_iter_sor = 200000
# when evaluating omega in the golden search we limit iterations to keep search fast
max_iter_eval = 50000

# (a) Vectorized cloud-in-cell (CIC)
def cloud_in_cell(xp, yp, M):
    #Vectorized CIC deposit to MxM grid. Particle coords assumed in [0, L).
    rho = np.zeros((M, M), dtype=float)

    # clamp particles to [0, L-eps] to avoid indexing to M
    eps = 1e-12
    xp = np.clip(xp, 0.0, L - eps)
    yp = np.clip(yp, 0.0, L - eps)

    # cell index (left / bottom) for each particle (cells are 0..M-1)
    i = np.floor(xp / dx).astype(int)  
    j = np.floor(yp / dx).astype(int)

    # local positions inside the cell, normalized [0,1)
    tx = (xp - i*dx) / dx
    ty = (yp - j*dx) / dx

    # bilinear weights for the four surrounding cells
    w00 = (1 - tx) * (1 - ty)
    w10 = tx * (1 - ty)
    w01 = (1 - tx) * ty
    w11 = tx * ty

    # neighbor indices (i+0,j+0), (i+1,j), (i,j+1), (i+1,j+1)
    i0, j0 = i, j
    i1, j1 = i + 1, j + 1

    # ensure indices in bounds
    valid00 = (i0 >= 0) & (i0 < M) & (j0 >= 0) & (j0 < M)
    valid10 = (i1 >= 0) & (i1 < M) & (j0 >= 0) & (j0 < M)
    valid01 = (i0 >= 0) & (i0 < M) & (j1 >= 0) & (j1 < M)
    valid11 = (i1 >= 0) & (i1 < M) & (j1 >= 0) & (j1 < M)

    # actual electron charge in coulombs
    q = -1.602176634e-19      # C



    # accumulate into rho using np.add.at for safe repeated indices
    if np.any(valid00):
        np.add.at(rho, (i0[valid00], j0[valid00]), q * w00[valid00])
    if np.any(valid10):
        np.add.at(rho, (i1[valid10], j0[valid10]), q * w10[valid10])
    if np.any(valid01):
        np.add.at(rho, (i0[valid01], j1[valid01]), q * w01[valid01])
    if np.any(valid11):
        np.add.at(rho, (i1[valid11], j1[valid11]), q * w11[valid11])

    return rho

rho = cloud_in_cell(x, y, M)

plt.figure(figsize=(6,5))
plt.imshow(rho.T, origin='lower', extent=[0,L,0,L])
plt.colorbar(label='charge density (arb)')
plt.title('Charge density (CIC)')
plt.savefig("rho.png", dpi=200)
plt.close()


# (b) Standard relaxation (Jacobi)
def relax_jacobi(phi0, rho, tol=target_tol, max_iter=max_iter_relax):
    phi_old = phi0.copy()
    # boundaries are grounded: phi_old[0,:] etc are zero and remain zero
    for it in range(1, max_iter+1):
        # compute new interior values using neighbor slices of phi_old
        phi_new = phi_old.copy()
        phi_new[1:-1,1:-1] = 0.25 * (
            phi_old[:-2,1:-1] + phi_old[2:,1:-1] +
            phi_old[1:-1,:-2] + phi_old[1:-1,2:] +
            h2 * rho[1:-1,1:-1] / epsilon0
        )
        diff = np.max(np.abs(phi_new - phi_old))
        if diff < tol:
            return phi_new, it
        phi_old = phi_new
    return phi_old, max_iter

# run Jacobi
phi0 = np.zeros((M, M), dtype=float)
t0 = time.time()
phi_relax, N_relax = relax_jacobi(phi0, rho)
t_elapsed = time.time() - t0
print(f"Jacobi relaxation converged in {N_relax} iterations, time {t_elapsed:.2f} s")

plt.figure(figsize=(6,5))
plt.imshow(phi_relax.T, origin='lower', extent=[0,L,0,L])
plt.colorbar(label='potential (arb)')
plt.title('Potential (Jacobi relaxation)')
plt.savefig("phi_relax.png", dpi=200)
plt.close()

# (c) Red-black SOR (Gauss-Seidel overrelaxation)
def sor_red_black(phi0, rho, omega, tol=target_tol, max_iter=max_iter_sor):
    phi = phi0.copy()
    I, J = np.indices(phi.shape)
    interior = np.zeros_like(phi, dtype=bool)
    interior[1:-1,1:-1] = True
    red_mask = ((I + J) % 2 == 0) & interior
    black_mask = ((I + J) % 2 == 1) & interior

    # rhs term (h2 * rho / epsilon0) for interior stored in full-size array
    rhs_full = np.zeros_like(phi)
    rhs_full[1:-1,1:-1] = (h2 * rho[1:-1,1:-1] / epsilon0)

    # For neighbor sums we need shifted arrays; we compute neighbors each sweep
    for it in range(1, max_iter+1):
        max_change = 0.0

        # compute neighbor sum for interior points using current phi
        neighbors = np.zeros_like(phi)
        neighbors[1:-1,1:-1] = (
            phi[:-2,1:-1] + phi[2:,1:-1] + phi[1:-1,:-2] + phi[1:-1,2:]
        )
        # the Gauss-Seidel target (un-relaxed) for interior is (neighbors + rhs)/4
        target = 0.25 * (neighbors + rhs_full)

        # Update red sites 
        old_red = phi[red_mask]
        phi[red_mask] = old_red + omega * (target[red_mask] - old_red)
        if old_red.size:
            change_red = np.max(np.abs(phi[red_mask] - old_red))
            if change_red > max_change: max_change = change_red

        # After red update, recompute neighbors and target for black (neighbors depend on red)
        neighbors[1:-1,1:-1] = (
            phi[:-2,1:-1] + phi[2:,1:-1] + phi[1:-1,:-2] + phi[1:-1,2:]
        )
        target = 0.25 * (neighbors + rhs_full)

        # Update black sites
        old_black = phi[black_mask]
        phi[black_mask] = old_black + omega * (target[black_mask] - old_black)
        if old_black.size:
            change_black = np.max(np.abs(phi[black_mask] - old_black))
            if change_black > max_change: max_change = change_black

        if max_change < tol:
            return phi, it
    return phi, max_iter

# quick test with theoretical guess for omega (good starting point)
omega_guess = 2.0 / (1.0 + sin(pi / M))
print(f"Omega theoretical initial guess: {omega_guess:.6f}")

t0 = time.time()
phi_sor_guess, its_guess = sor_red_black(np.zeros((M,M)), rho, omega_guess, tol=target_tol, max_iter=20000)
t_elapsed = time.time() - t0
print(f"SOR with guess omega {omega_guess:.6f} converged in {its_guess} iterations, time {t_elapsed:.2f} s")


# Golden-section search for optimal omega (precision 0.001)
# Search in (1.0, 1.99). omega must be <2.
# Each objective evaluation runs SOR but limited to max_iter_eval iterations for speed.
def objective_iters(omega):
    phi0 = np.zeros((M, M), dtype=float)
    _, iters = sor_red_black(phi0, rho, omega, tol=target_tol, max_iter=max_iter_eval)
    if iters >= max_iter_eval:
        return 1_000_000  # penalty for not converging fast enough
    return iters

# golden-section setup
a, b = 1.0, 1.99
gr = (np.sqrt(5) - 1) / 2.0
c = b - gr * (b - a)
d = a + gr * (b - a)

history = []   # will store tuples (a,b,c,d, f_c, f_d)

# evaluate initial points
f_c = objective_iters(c)
f_d = objective_iters(d)
history.append((a, b, c, d, f_c, f_d))

# iterate until bracket width < 0.001
while (b - a) > 0.001:
    if f_c < f_d:
        # keep [a, d]
        b = d
        d = c
        f_d = f_c
        c = b - gr * (b - a)
        f_c = objective_iters(c)
    else:
        # keep [c, b]
        a = c
        c = d
        f_c = f_d
        d = a + gr * (b - a)
        f_d = objective_iters(d)
    history.append((a, b, c, d, f_c, f_d))
    # Safety break 
    if len(history) > 200:
        break

omega_opt = 0.5 * (a + b)
print(f"Golden-section search finished. omega_opt ~ {omega_opt:.6f}")

# extract history for plotting omega evolution  (plot mid-point of bracket each step)
omega_steps = [0.5*(h[0]+h[1]) for h in history]
iters_steps = [min(h[4], h[5]) for h in history]

plt.figure(figsize=(6,4))
plt.plot(omega_steps, marker='o')
plt.xlabel('Golden search step')
plt.ylabel('Bracket midpoint ω')
plt.title('ω evolution during golden-section search')
plt.grid(True)
plt.savefig("omega_search.png", dpi=200)
plt.close()

# Solve with best omega found (full convergence)
t0 = time.time()
phi_sor_opt, N_sor = sor_red_black(np.zeros((M,M)), rho, omega_opt, tol=target_tol, max_iter=max_iter_sor)
t_elapsed = time.time() - t0
print(f"SOR with omega_opt {omega_opt:.6f} converged in {N_sor} iterations, time {t_elapsed:.2f} s")

plt.figure(figsize=(6,5))
plt.imshow(phi_sor_opt.T, origin='lower', extent=[0,L,0,L])
plt.colorbar(label='potential (arb)')
plt.title(f'Potential (SOR, ω={omega_opt:.3f})')
plt.savefig("phi_sor.png", dpi=200)
plt.close()

# summary print
print("Summary:")
print(f"  Jacobi iterations: {N_relax}")
print(f"  SOR (guess ω={omega_guess:.6f}) iterations: {its_guess}")
print(f"  SOR (opt ω={omega_opt:.6f}) iterations: {N_sor}")
print(f"  Golden search steps: {len(history)}")
