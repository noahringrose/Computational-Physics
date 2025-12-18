import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt, cos, sin, pi, exp

# (a) Linear congruential generator (LCG) for U[0,1)
class LCG:
    """
    LCG: X_{n+1} = (a X_n + c) mod m
    U_n = X_n / m  in [0,1).
    """
    def __init__(self, seed=123456789, a=1664525, c=1013904223, m=2**32):
        self.a = int(a)
        self.c = int(c)
        self.m = int(m)
        self.state = int(seed) % self.m

    def rand_uint(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def rand(self):
        return self.rand_uint() / self.m  # float in [0,1)

    def rand_array(self, n):
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = self.rand()
        return out


# (b) Gaussian generator 
    
def gaussian_box_muller(lcg: LCG, n: int):
    """
    Generate n iid N(0,1) values using Box–Muller from LCG uniforms.
    """
    out = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        # Avoid log(0) by shifting u1 away from 0
        u1 = max(lcg.rand(), 1e-12)
        u2 = lcg.rand()

        r = sqrt(-2.0 * log(u1))
        theta = 2.0 * pi * u2
        z0 = r * cos(theta)
        z1 = r * sin(theta)

        out[i] = z0
        i += 1
        if i < n:
            out[i] = z1
            i += 1
    return out


# (c,e) Power spectrum via FFT (power = |FT|^2)

def power_spectrum(x):
    """
    Return (k, Pk) for a real 1D signal x of length N.
    Uses rfft -> k=0..N/2. Ignores k=0 in slope fits/plots if desired.
    """
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    Xk = np.fft.rfft(x)
    Pk = (np.abs(Xk)**2) / (N**2)  # consistent normalization for comparisons
    k = np.arange(Pk.size)         # integer wavenumber index
    return k, Pk


def fit_loglog_slope(k, Pk, kmin=1, kmax=None):
    """
    Fit slope of log P vs log k over k in [kmin, kmax].
    Returns slope, intercept. Excludes any nonpositive entries.
    """
    if kmax is None:
        kmax = k[-1]

    mask = (k >= kmin) & (k <= kmax) & (Pk > 0)
    kk = k[mask]
    pp = Pk[mask]
    logk = np.log(kk)
    logp = np.log(pp)
    slope, intercept = np.polyfit(logk, logp, 1)
    return slope, intercept



def main():
    # Problem specs
    N = 10_000
    seed = 123456789

    # (a) Choose LCG constants 
    a = 1664525
    c = 1013904223
    m = 2**32
    lcg = LCG(seed=seed, a=a, c=c, m=m)

    # Generate uniforms + Gaussians
    gauss = gaussian_box_muller(lcg, N)

    # (b) Histogram vs unit Gaussian PDF, y-axis log scale
    plt.figure()
    bins = 80
    counts, edges, _ = plt.hist(gauss, bins=bins, density=True, alpha=0.6, label="Generated (N=10,000)")
    xs = np.linspace(edges[0], edges[-1], 600)
    pdf = (1.0 / np.sqrt(2*np.pi)) * np.exp(-0.5 * xs**2)
    plt.plot(xs, pdf, linewidth=2, label="Unit Gaussian PDF")
    plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel("Probability density (log scale)")
    plt.title("Gaussian RNG via Box–Muller (from LCG uniforms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gaussian_hist_logy.png", dpi=200)
    plt.close()

    # (c) Power spectrum of Gaussian noise
    k_g, P_g = power_spectrum(gauss)

    # Avoid k=0 in log-log plotting
    plt.figure()
    plt.loglog(k_g[1:], P_g[1:], linewidth=1)
    plt.xlabel("Wavenumber k")
    plt.ylabel("Power P(k)")
    plt.title("Power spectrum of Gaussian white noise (log-log)")
    plt.tight_layout()
    plt.savefig("power_spectrum_gaussian_loglog.png", dpi=200)
    plt.close()

    slope_g, _ = fit_loglog_slope(k_g, P_g, kmin=5, kmax=min(2000, k_g[-1]))

    # (d) Random walk from Gaussian steps
    walk = np.cumsum(gauss)

    plt.figure()
    plt.plot(np.arange(N), walk, linewidth=1)
    plt.xlabel("Iteration i")
    plt.ylabel("x(i)")
    plt.title("Random walk generated from Gaussian steps")
    plt.tight_layout()
    plt.savefig("random_walk.png", dpi=200)
    plt.close()

    # (e) Power spectrum of random walk
    k_w, P_w = power_spectrum(walk)

    plt.figure()
    plt.loglog(k_w[1:], P_w[1:], linewidth=1, label="Random walk power")
    # Reference k^{-2} line (scaled to match roughly)
    ref_k = k_w[1:].astype(float)
    # choose scale using one mid-point
    mid = min(500, ref_k.size - 1)
    C = P_w[1:][mid] * (ref_k[mid]**2)
    plt.loglog(ref_k, C * ref_k**(-2), linewidth=2, label=r"Reference $\propto k^{-2}$")
    plt.xlabel("Wavenumber k")
    plt.ylabel("Power P(k)")
    plt.title("Power spectrum of random walk (log-log)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("power_spectrum_walk_loglog.png", dpi=200)
    plt.close()

    slope_w, _ = fit_loglog_slope(k_w, P_w, kmin=5, kmax=min(2000, k_w[-1]))

    with open("slope_summary.tex", "w") as f:
        f.write("\\begin{tabular}{lcc}\\hline\n")
        f.write("Signal & Expected slope & Fitted slope \\\\\\hline\n")
        f.write(f"Gaussian noise & 0 & {slope_g:.2f} \\\\\n")
        f.write(f"Random walk & $-2$ & {slope_w:.2f} \\\\\n")
        f.write("\\hline\\end{tabular}\n")

    print("Saved figures:")
    print("  gaussian_hist_logy.png")
    print("  power_spectrum_gaussian_loglog.png")
    print("  random_walk.png")
    print("  power_spectrum_walk_loglog.png")
    print("Also wrote:")
    print("  slope_summary.tex")
    print("\nLCG parameters used:")
    print(f"  m = {m}, a = {a}, c = {c}, seed = {seed}")
    print(f"Fitted slopes (log P vs log k over k ~ 5..2000):")
    print(f"  Gaussian noise slope ~ {slope_g:.2f} (expected ~ 0)")
    print(f"  Random walk slope   ~ {slope_w:.2f} (expected ~ -2)")


if __name__ == "__main__":
    main()
