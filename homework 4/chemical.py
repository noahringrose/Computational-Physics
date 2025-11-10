"""
Brusselator with adaptive Bulirsch–Stoer (modified-midpoint base).
- Starts from a single interval H=20 and recursively subdivides on failure.
- Enforces |error|_component <= delta*H  (absolute per-unit-time), with a tiny relative cushion.
"""

import numpy as np
import matplotlib.pyplot as plt

# Problem setup
a = 1.0
b = 3.0
t0, t1 = 0.0, 20.0
y0 = np.array([0.0, 0.0], dtype=float)

delta = 1e-10           # absolute per-unit-time target 
rtol  = 1e-9            # tiny relative cushion (prevents pathological over-refinement)

H_initial = 20.0        # start with one giant step; split if needed
nmax = 24               # allow deeper extrapolation: 2,4,...,24  (crucial for δ=1e-10)

# Safety + runtime controls
ABORT_MAG = 1e5         # if any intermediate |x| or |y| exceeds this => reject step immediately
H_MIN = 1e-12           # minimum step size before we declare failure
MAX_ITERS = 400000      # hard cap on driver iterations

#  The Brusselator
def f(t, y):
    x, yv = y
    dx = 1.0 - (b + 1.0) * x + a * x * x * yv
    dy = b * x - a * x * x * yv
    return np.array([dx, dy])

def finite_and_reasonable(v):
    return np.all(np.isfinite(v)) and np.max(np.abs(v)) < ABORT_MAG

# Modified-midpoint base method on [t, t+H] with n substeps (even)
# Returns (y_end, ok_flag). If anything goes non-finite or huge, returns (None, False).

def modified_midpoint(y, t, H, n):
    h = H / n
    y0 = y.copy()
    if not finite_and_reasonable(y0):
        return None, False

    # first step
    y1 = y0 + h * f(t, y0)
    if not finite_and_reasonable(y1):
        return None, False

    tc = t + h
    for _ in range(2, n + 1):
        y2 = y0 + 2.0 * h * f(tc, y1)
        if not finite_and_reasonable(y2):
            return None, False
        y0, y1 = y1, y2
        tc += h

    corr = h * f(tc, y1)
    if not finite_and_reasonable(corr):
        return None, False

    y_end = 0.5 * (y1 + y0 + corr)
    if not finite_and_reasonable(y_end):
        return None, False

    return y_end, True

# One Bulirsch–Stoer attempt across H
# - Base: modified-midpoint with n in 2,4,...,nmax
# - Extrapolation: Neville in s=(H/n)^2
# - Accept if componentwise: |err| <= delta*H + (rtol * max(|y_new|,|y_old|))

def bs_attempt(y, t, H, delta, rtol, nmax):
    nseq = list(range(2, nmax + 1, 2))   # 2,4,...,nmax
    diag = []                            # Neville diagonal (vector-valued)

    for k, n in enumerate(nseq):
        y_mm, ok = modified_midpoint(y, t, H, n)
        if not ok:
            return False, None, np.inf   # base blew up; reject immediately

        if k == 0:
            diag.append(y_mm)
        else:
            s_k = (H / nseq[k])**2
            cur = y_mm.copy()
            # Build current diagonal from previous entries (backwards)
            for j in range(k - 1, -1, -1):
                s_j = (H / nseq[j])**2
                factor = s_j / (s_j - s_k)
                cur = cur + (cur - diag[j]) * factor
                if not finite_and_reasonable(cur):
                    return False, None, np.inf
            diag.append(cur)

        # From second diagonal onward we can estimate error
        if k >= 1:
            diff = diag[-1] - diag[-2]
            err_vec = np.abs(diff)
            scale  = rtol * np.maximum(np.abs(diag[-1]), np.abs(diag[-2]))
            # enforce absolute-per-unit-time AND tiny relative cushion
            ok_comp = err_vec <= (delta * H + scale)
            if np.all(ok_comp):
                return True, diag[-1], float(np.max(err_vec))

    # Not accurate enough at max n
    if len(diag) >= 2 and np.all(np.isfinite(diag[-1])) and np.all(np.isfinite(diag[-2])):
        err = float(np.max(np.abs(diag[-1] - diag[-2])))
    else:
        err = np.inf
    return False, (diag[-1] if len(diag) else None), err

# Recursive (stack-based) driver:
# - Start with H=20 across [0,20]; split in half on failure; repeat.
# - Record accepted boundaries for dots on the plot.
def integrate_recursive(y0, t0, t1, H_initial, delta, rtol, nmax):
    stack = [(t0, y0.copy(), H_initial)]
    ts = [t0]
    ys = [y0.copy()]
    dots = [t0]
    known = {t0: y0.copy()}

    iters = 0
    while stack:
        iters += 1
        if iters > MAX_ITERS:
            raise RuntimeError("Exceeded MAX_ITERS; aborting to avoid infinite loop.")

        t, y, H = stack.pop()
        if t >= t1:
            continue
        if t + H > t1:
            H = t1 - t

        ok, y_end, err = bs_attempt(y, t, H, delta, rtol, nmax)
        print(f"[{iters:06d}] t={t:.6f}, H={H:.3e}, err={err:.3e}, accepted={ok}")

        if ok:
            tE = t + H
            ts.append(tE)
            ys.append(y_end.copy())
            dots.append(tE)
            known[tE] = y_end.copy()
            # If top stack item needs this starting y (second half placeholder), fill it
            if stack and stack[-1][1] is None:
                t_need, _, H_need = stack[-1]
                if t_need in known:
                    stack[-1] = (t_need, known[t_need], H_need)
        else:
            if H <= H_MIN:
                raise RuntimeError("Step size underflow while enforcing tolerance.")
            h2 = 0.5 * H
            t_mid = t + h2
            print(f"          -> reject; halve H to {h2:.3e}")
            # push second half first so first half is processed next
            stack.append((t_mid, None, h2))
            stack.append((t, y, h2))

    # sort and dedup
    order = np.argsort(ts)
    ts = np.array(ts)[order]
    ys = np.array(ys)[order]
    keep_t = [ts[0]]
    keep_y = [ys[0]]
    for i in range(1, len(ts)):
        if abs(ts[i] - keep_t[-1]) > 1e-14:
            keep_t.append(ts[i])
            keep_y.append(ys[i])

    return np.array(keep_t), np.array(keep_y), np.unique(np.array(dots))

# Main: run, plot, save
if __name__ == "__main__":
    t, Y, dots = integrate_recursive(y0, t0, t1, H_initial, delta, rtol, nmax)
    x = Y[:, 0]
    y = Y[:, 1]

    # Time series with accepted boundaries
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, x, label="x(t)")
    plt.plot(t, y, label="y(t)")

    # show dots only on x(t) to reduce clutter
    xd = np.interp(dots, t, x)

    # subsample so we draw at most ~200 dots
    max_dots = 200
    stride = max(1, len(dots) // max_dots)
    dots_s = dots[::stride]
    xd_s   = xd[::stride]

    plt.plot(dots_s, xd_s, 'o', ms=5.5, mfc='white', mec='k', mew=0.9, zorder=6, label="step boundaries")
    plt.legend()

    plt.xlabel("t")
    plt.ylabel("Concentration")
    plt.title("Brusselator: adaptive Bulirsch–Stoer (δ=1e-10 per unit time)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("brusselator_timeseries.png", dpi=180)

    # Phase portrait
    plt.figure(figsize=(5.2, 5.2))
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Brusselator Phase Portrait")
    plt.tight_layout()
    plt.savefig("brusselator_phase.png", dpi=180)

    # Data dump
    np.savetxt("brusselator_solution.txt",
               np.column_stack([t, x, y]),
               header="t   x(t)   y(t)")
