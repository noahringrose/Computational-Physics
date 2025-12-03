import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Physical parameters
hbar = 1.055e-34        # Planck's constant (J s)
M    = 9.109e-31        # electron mass (kg)
L    = 1.0e-8           # box length (m)

# Numerical parameters 
N    = 1000             # number of spatial intervals (N+1 grid points)
a    = L / N            # spatial grid spacing
h    = 1.0e-18          # time step (s)
nsteps = 2000           # number of time steps to evolve

# Spatial grid: interior points only (Dirichlet boundaries at 0 and L)
x = np.linspace(a, L - a, N - 1)

# Initial condition: Gaussian wave packet
x0    = L / 2.0
sigma = 1.0e-10
kappa = 5.0e10

psi = np.exp(-(x - x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kappa * x)

# normalize
norm = np.sqrt(np.sum(np.abs(psi)**2) * a)
psi /= norm

# Build Crank–Nicolson matrices 
gamma = 1j * hbar * h / (2.0 * M * a**2)

a1 = 1.0 + gamma      # main diag of A
a2 = -0.5 * gamma     # off diag of A
b1 = 1.0 - gamma      # main diag of B
b2 = 0.5 * gamma      # off diag of B

diag_main_A = a1 * np.ones(N - 1, dtype=complex)
diag_off_A  = a2 * np.ones(N - 2, dtype=complex)

diag_main_B = b1 * np.ones(N - 1, dtype=complex)
diag_off_B  = b2 * np.ones(N - 2, dtype=complex)

A = np.diag(diag_main_A) + \
    np.diag(diag_off_A,  1) + \
    np.diag(diag_off_A, -1)

B = np.diag(diag_main_B) + \
    np.diag(diag_off_B,  1) + \
    np.diag(diag_off_B, -1)


def crank_nicolson_step(psi, A, B):
    """Perform a single Crank–Nicolson time step."""
    v = B @ psi
    psi_new = np.linalg.solve(A, v)
    return psi_new


def evolve(psi0, A, B, nsteps):
    """Evolve the wavefunction for nsteps time steps."""
    psi = psi0.copy()
    history = [psi.copy()]
    for _ in range(nsteps):
        psi = crank_nicolson_step(psi, A, B)
        history.append(psi.copy())
    return np.array(history)


#  Time evolution 
psi_history = evolve(psi, A, B, nsteps)

#  Animation of Re[psi] 
fig, ax = plt.subplots()
line, = ax.plot(x, np.real(psi_history[0]), lw=2)

ax.set_xlim(0, L)
ymax = 1.2 * np.max(np.abs(np.real(psi_history)))
ax.set_ylim(-ymax, ymax)
ax.set_xlabel("x (m)")
ax.set_ylabel("Re[psi(x,t)]")
ax.set_title("Time evolution of Re[psi(x,t)]")


def update(frame):
    line.set_ydata(np.real(psi_history[frame]))
    t_fs = frame * h * 1e15  # convert to femtoseconds
    ax.set_title(f"Re[psi(x,t)], step = {frame},  t = {t_fs:.3f} fs")
    return line,  

ani = FuncAnimation(fig, update, frames=len(psi_history),
                    interval=10, blit=False)


plt.show()
