"""
Exercise 7.2 — Sunspot Periodicity
"""
from __future__ import annotations
import sys
from pathlib import Path

# Was having issues getting Python to read the file and load the plots. Ignore this code below. 
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (no GUI required)
    import matplotlib.pyplot as plt
except Exception as e:
    msg = (
        "Matplotlib is not available in this Python environment.\n"
        "Quick fix:\n"
        "  • If you are in a virtualenv:    python -m pip install matplotlib numpy\n"
        "  • If you are NOT in a venv:      python3 -m venv .venv && "
        "source .venv/bin/activate && python -m pip install matplotlib numpy\n"
        "Then re-run: python sunspots_periodicity.py\n\n"
        f"Details: {e}"
    )
    raise SystemExit(msg)

import numpy as np


def here() -> Path:
    """Directory where this script lives."""
    return Path(__file__).resolve().parent
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Everything sorted out now. Real code starts here.

def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find data file at:\n  {path}\n"
            "Make sure sunspots.txt is in the SAME folder as this script."
        )
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(
            f"Expected two columns in {path.name} (month, sunspots). "
            f"Got array shape {arr.shape}."
        )
    months = arr[:, 0].astype(int)
    spots = arr[:, 1].astype(float)
    return months, spots


def plot_time_series(months: np.ndarray, spots: np.ndarray, outpath: Path) -> None:
    plt.figure(figsize=(9, 3.5))
    plt.plot(months, spots, linewidth=1)
    plt.xlabel("Month since Jan 1749")
    plt.ylabel("Sunspot number")
    plt.title("Monthly Sunspot Numbers")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def compute_power_spectrum(spots: np.ndarray, dt: float = 1.0):
    x = spots - np.mean(spots) #Recenters data on y axis
    N = len(x)
    X = np.fft.rfft(x)                  # one-sided FFT using Python packages
    power = (np.abs(X) ** 2) / (N ** 2)
    freqs = np.fft.rfftfreq(N, d=dt)    # cycles per month
    k_vals = np.arange(len(power))
    return freqs, power, k_vals, N


def plot_power_vs_k(k_vals, power, outpath: Path, kmax: int | None = None) -> None:
    plt.figure(figsize=(8, 3.5))
    plt.semilogy(k_vals, power, linewidth=1)
    if kmax is None:
        kmax = min(200, len(k_vals) - 1)
    plt.xlim(0, kmax)
    plt.xlabel("Fourier mode index k")
    plt.ylabel(r"$|c_k|^2$ (power)")
    plt.title("Power Spectrum of Sunspot Signal (one-sided)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_power_vs_freq(freqs, power, outpath: Path, fmax: float = 0.05) -> None:
    plt.figure(figsize=(8, 3.5))
    plt.plot(freqs, power, linewidth=1)
    plt.xlim(0, fmax)  # emphasize long periods (low frequency)
    plt.xlabel("Frequency (cycles per month)")
    plt.ylabel(r"$|c_k|^2$ (power)")
    plt.title("Power Spectrum vs Frequency")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main() -> int:
    base = here()
    data_path = base / "sunspots.txt"
    out_ts = base / "sunspots_timeseries.png"
    out_psk = base / "sunspots_power_spectrum_k.png"
    out_psf = base / "sunspots_power_spectrum_f.png"

    print("=== Sunspot Periodicity Analysis ===")
    print(f"Script directory: {base}")
    print(f"Reading data file: {data_path}")

    months, spots = load_data(data_path)

    # (a) time series
    plot_time_series(months, spots, out_ts)

    # (b) power spectrum
    freqs, power, k_vals, N = compute_power_spectrum(spots, dt=1.0)

    # --- Find dominant nonzero peak
    k_peak = int(np.argmax(power[1:]) + 1)   # ignore k=0 and return k with largest power
    f_peak = float(freqs[k_peak])            # cycles/month
    T_months = float(1.0 / f_peak)
    T_years = T_months / 12.0


    # --- Plot the power spectrum (trim low-k and compress y-scale)
    plt.figure(figsize=(8, 3.5))

    skip = 10
    plt.semilogy(k_vals[skip:150], power[skip:150], linewidth=1.6, color='navy')

    # Highlight the dominant peak
    plt.axvline(k_peak, color='r', linestyle='--', linewidth=1.2)

    # Y-axis limits 
    plt.ylim(1e-1, 3e2)    # spans roughly from 0.01 to 300

    plt.xlabel("Fourier mode index k")
    plt.ylabel(r"$|c_k|^2$ (power)")
    plt.title("Power Spectrum (first 150 modes, low-k region)")
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("sunspots_power_spectrum_k.png", dpi=200)
    plt.close()



    print(f"k* = {k_peak}, f* = {f_peak:.6f} cycles/mo, T = {T_months:.2f} months")


    # plots
    plot_power_vs_k(k_vals, power, out_psk)
    plot_power_vs_freq(freqs, power, out_psf)

    # report
    print(f"N (months):         {N}")
    print(f"Peak mode k*:       {k_peak}")
    print(f"Frequency f*:       {f_peak:.9f} cycles/month")
    print(f"Period T:           {T_months:.6f} months  (~{T_years:.4f} years)")
    print("Saved figures:")
    print(f"  - {out_ts}")
    print(f"  - {out_psk}")
    print(f"  - {out_psf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
