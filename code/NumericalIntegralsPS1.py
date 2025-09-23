import numpy as np
import matplotlib.pyplot as plt

# Function and exact integral
def f(t): return np.float32(np.exp(-t))
I_exact = 1 - np.exp(-1)  # analytic result

# Midpoint rule
def midpoint_rule(f, a, b, N):
    h = np.float32((b - a) / N)
    s = np.float32(0.0)
    for i in range(N):
        x_mid = a + (i + 0.5) * h
        s += f(x_mid)
    return h * s

# Trapezoid rule
def trapezoid_rule(f, a, b, N):
    h = np.float32((b - a) / N)
    s = (f(a) + f(b)) / 2.0
    for i in range(1, N):
        s += f(a + i*h)
    return h * s

# Simpson's rule (N must be even)
def simpson_rule(f, a, b, N):
    if N % 2 == 1:
        N += 1  # enforce even
    h = np.float32((b - a) / N)
    s = f(a) + f(b)
    for i in range(1, N, 2):
        s += 4 * f(a + i*h)
    for i in range(2, N, 2):
        s += 2 * f(a + i*h)
    return (h/3.0) * s

methods = {
    "Midpoint": midpoint_rule,
    "Trapezoid": trapezoid_rule,
    "Simpson": simpson_rule
}

a, b = 0.0, 1.0
N_values = np.logspace(1, 7, 50, dtype=int)  # from ~10 bins to ~10^7

plt.figure(figsize=(6,4))
for name, method in methods.items():
    errors = []
    for N in N_values:
        approx = method(f, a, b, N)
        rel_err = abs((approx - I_exact) / I_exact)
        errors.append(rel_err)
    plt.loglog(N_values, errors, label=name)

plt.xlabel("Number of bins N")
plt.ylabel("Relative error ε")
plt.title("Integration of exp(-t) from 0 to 1")
plt.legend()
plt.grid(True, which="both")
plt.savefig("integration_errors.png", dpi=150)