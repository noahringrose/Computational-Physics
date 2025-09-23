import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


#Define Simpson Integration
def simpson(f, a, b, N):
    if N % 2 == 1:
        N += 1  # enforce even
    h = np.float32((b - a) / N)
    s = f(a) + f(b)
    for i in range(1, N, 2):
        s += 4 * f(a + i*h)
    for i in range(2, N, 2):
        s += 2 * f(a + i*h)
    return (h/3.0) * s

# Load the tabulated power spectrum (k, P(k))
data = np.loadtxt("lcdm_z0.matter_pk")
k_data = data[:,0]   # k in h/Mpc
Pk_data = data[:,1]  # P(k)


# Interpolate log-log since P(k) ~ power law in k
Pk_interp = interp1d(np.log(k_data), np.log(Pk_data), kind='cubic', fill_value="extrapolate")

def Pk(k):
    return np.exp(Pk_interp(np.log(k)))

# Correlation function xi(r)
def xi_of_r(r, kmin=1e-4, kmax=1000.0, npts=200000):
    k = np.logspace(np.log10(kmin), np.log10(kmax), npts)
    def integrand(k):
        return k**2 * Pk(k) * np.sinc(k*r/np.pi)  
    I = simpson(integrand, kmin, kmax, npts)
    return I / (2*np.pi**2)

# Compute xi(r) over desired range
r_vals = np.linspace(50, 120, 200)  # Mpc/h
xi_vals = np.array([xi_of_r(r) for r in r_vals])

# Multiply by r^2 to emphasize BAO bump
y_vals = r_vals**2 * xi_vals

# Find the BAO peak
restrictedrange = (r_vals > 85) & (r_vals < 120)
peak_idx = np.argmax(y_vals[restrictedrange])
r_peak = r_vals[restrictedrange][peak_idx]


print(f"BAO peak located at r ≈ {r_peak:.2f} Mpc/h")

# Plot
plt.figure(figsize=(6,4))
plt.plot(r_vals, y_vals, label=r"$r^2 \xi(r)$")
plt.axvline(r_peak, color='r', ls='--', label=f"BAO peak ~ {r_peak:.1f} Mpc/h")
plt.xlabel("r [Mpc/h]")
plt.ylabel(r"$r^2 \xi(r)$")
plt.title("Baryon Acoustic Oscillation (BAO) feature")
plt.legend()
plt.grid(True)
plt.savefig("bao_peak.png", dpi=150)