"""
Exercise 7.9 — Deconvolution of Image
"""

from __future__ import annotations
import os
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Stuff to ensure things land in the right folder and so
HERE = Path(__file__).resolve().parent
os.chdir(HERE)

# (a) Load the image as array, then invert the array as image was initially upside-down
def load_blur(path: Path) -> np.ndarray:
    """Load blur.txt as 2D array (float)."""
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")
    return np.flipud(arr)


# (b) Gaussian Point Spread Function
def gaussian_psf(shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Centered Gaussian on periodic grid (bright corners when plotted)."""
    K, L = shape
    # coordinate arrays centered at (0,0) in a periodic sense
    ky = np.arange(K)
    kx = np.arange(L)
    # center indices (so 0 is the "origin" for periodic convolution)
    y = (ky - K * (ky > K // 2))  # 0,1,2,...,K/2, -(K/2-1),..., -1
    x = (kx - L * (kx > L // 2))
    Y, X = np.meshgrid(y, x, indexing="ij")
    psf = np.exp(-(X**2 + Y**2) / (2.0 * sigma**2))
    psf /= psf.sum()  # unit DC gain
    return psf

#(c) The whole shebang
def deconvolve_fft(blur: np.ndarray, psf: np.ndarray, eps: float) -> np.ndarray:
    """FFT deconvolution with threshold: if |F_psf|<eps, leave coefficient alone."""
    #find size of lattice
    K, L = blur.shape
    #find fourier coefficients of the blurred image brightness
    B = np.fft.rfft2(blur)
    #find fourier coefficients of the convolution function
    F = np.fft.rfft2(psf)
    A = np.empty_like(B, dtype=np.complex128)

    mag = np.abs(F)
    mask = mag >= eps
    #Fourier coefficients of new image
    A[mask] = B[mask] / F[mask]
    A[~mask] = B[~mask]  # leave as-is when PSF transfer is tiny
    #Inverse transform to get unblurred image 
    recon = np.fft.irfft2(A, s=(K, L))
    return recon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=HERE / "blur.txt",
                    help="Path to blur.txt (2D grayscale array).")
    ap.add_argument("--sigma", type=float, default=25.0,
                    help="Gaussian PSF width (pixels).")
    ap.add_argument("--eps", type=float, default=1e-3,
                    help="Threshold for |FFT(PSF)| to avoid division by ~0.")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"ERROR: {args.input} not found")

    # (a) load and show blurred image
    b = load_blur(args.input)
    K, L = b.shape
    plt.figure(figsize=(6.4, 6.0))
    plt.imshow(b, cmap="gray", origin="lower", aspect="equal")
    plt.title("Blurred image")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("blur_image.png", dpi=220)
    plt.close()

    # (b) build and show PSF
    psf = gaussian_psf((K, L), sigma=args.sigma)
    plt.figure(figsize=(6.4, 6.0))
    plt.imshow(psf, cmap="gray", origin="lower", aspect="equal")
    plt.title(f"Gaussian PSF (sigma={args.sigma:g})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("psf_sigma25.png", dpi=220)
    plt.close()

    # (c) FFT deconvolution with epsilon safeguard
    a_rec = deconvolve_fft(b, psf, eps=args.eps)
    plt.figure(figsize=(6.4, 6.0))
    plt.imshow(a_rec, cmap="gray", origin="lower", aspect="equal")
    plt.title(f"Deconvolved image (sigma={args.sigma:g}, eps={args.eps:g})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"deblur_sigma{int(args.sigma)}_eps{args.eps:.0e}.png", dpi=220)
    plt.close()

    # tiny console summary
    print(f"Loaded blur.txt with shape {K}x{L}")
    print(f"PSF sigma={args.sigma}, eps={args.eps}")
    print("Saved: blur_image.png, psf_sigma25.png, deblur_*.png")


if __name__ == "__main__":
    main()
