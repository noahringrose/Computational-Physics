"""
Exercise 7.4 — Fourier Filtering of the DJIA Series
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- ensure saving in homework3 folder
HERE = Path(__file__).resolve().parent
os.chdir(HERE)

# --- load data
y = np.loadtxt("dow.txt", dtype=float)
N = len(y)
t = np.arange(N)

# --- (a) raw time series
plt.figure(figsize=(10, 4))
plt.plot(t, y, lw=1.2)
plt.xlabel("Trading day index")
plt.ylabel("DJIA close")
plt.title("DJIA daily close (business days, 2006–2010)")
plt.tight_layout()
plt.savefig("dow_timeseries.png", dpi=220)
plt.close()

# --- helper: low-pass filter by fraction
def lowpass_fraction(x, frac):
    X = np.fft.rfft(x)
    M = X.size
    K = int(np.ceil(frac * M))
    Xf = np.zeros_like(X)
    Xf[:K] = X[:K]
    xrec = np.fft.irfft(Xf, n=len(x))
    return xrec, K, M

# --- (b)-(d) keep first 10% of modes
y10, K10, M = lowpass_fraction(y, 0.10)
plt.figure(figsize=(10, 4))
plt.plot(t, y, lw=0.9, label="Original", color="C0")
plt.plot(t, y10, lw=1.6, label="Low-pass (10%)", color="C1")
plt.xlabel("Trading day index")
plt.ylabel("DJIA close")
plt.title("Fourier low-pass reconstruction (keep 10% of modes)")
plt.legend()
plt.tight_layout()
plt.savefig("dow_filtered_10pct.png", dpi=220)
plt.close()

# --- (e) keep first 2% of modes
y02, K02, _ = lowpass_fraction(y, 0.02)
plt.figure(figsize=(10, 4))
plt.plot(t, y, lw=0.9, label="Original", color="C0")
plt.plot(t, y02, lw=1.6, label="Low-pass (2%)", color="C2")
plt.xlabel("Trading day index")
plt.ylabel("DJIA close")
plt.title("Fourier low-pass reconstruction (keep 2% of modes)")
plt.legend()
plt.tight_layout()
plt.savefig("dow_filtered_2pct.png", dpi=220)
plt.close()

print(f"N={N}, rFFT length M={M}, K10={K10}, K02={K02}")
print("Saved: dow_timeseries.png, dow_filtered_10pct.png, dow_filtered_2pct.png")
