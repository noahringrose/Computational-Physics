# wien_bisection.py
import math
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 5.0*math.exp(-x) + x - 5.0

# interval
a = 1e-6
b = 20.0
# ensure the endpoints satisfy f(a)f(b)<0
while f(a)*f(b) > 0:
    b *= 2.0
    if b > 1e6:
        raise RuntimeError("Cannot find bracket")

tol = 1e-6
iters = 0
while (b - a)/2.0 > tol:
    c = 0.5*(a + b)
    if f(c) == 0:
        a = b = c
        break
    if f(a)*f(c) < 0:
        b = c
    else:
        a = c
    iters += 1

#root is approximately equal to midpoint of interval
x = 0.5*(a + b)
print(f"Root x ≈ {x:.9f}  (iterations = {iters})")
# physical constants
h = 6.62607015e-34   # J s
c = 299792458.0      # m/s
kB = 1.380649e-23    # J/K

b_wien = h * c / (kB * x)     # meters * Kelvin
print(f"Wien constant b ≈ {b_wien:.12e} m K  = {b_wien*1e6:.9f} µm K")

# Sun temperature from lambda_peak = 502 nm
lambda_sun = 502e-9
T_sun = b_wien / lambda_sun
print(f"Estimated Sun temperature T ≈ {T_sun:.1f} K")

#plot f(x) and mark root
xs = np.linspace(0.01, 10, 400)
ys = [5*np.exp(-xx) + xx - 5 for xx in xs]
plt.figure(figsize=(6,4))
plt.plot(xs, ys, label='f(x)=5 e^{-x} + x - 5')
plt.axhline(0, color='k', lw=0.6)
plt.axvline(x, color='r', ls='--', label=f'root x={x:.6f}')
plt.xlim(0, 10)
plt.ylim(min(ys), max(ys))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.tight_layout()
plt.show()
