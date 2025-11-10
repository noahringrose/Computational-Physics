
"""
Supermassive BH binary (equal masses -> one-body reduction).
Units: G = M_BH = 1, distance unit = 100 pc, Schwarzschild radius r_s = 1e-7.
Equation for one BH: r¨ = -(μ r)/r^3 + a_DF,   μ = GM/4
Dynamical friction:  a_DF = -(A/(|v|^3 + B)) v

This script produces all figures & CSVs for parts (a)–(d) in ./out/.
It is tuned to finish in a reasonable time on my Macbook.
"""

import os, math, numpy as np
import matplotlib.pyplot as plt

# Global parameters / numerics

OUT = "out"; os.makedirs(OUT, exist_ok=True)

G, MBH = 1.0, 1.0
mu = G * MBH / 4.0
r_s = 1e-7

# Adaptive RK4 knobs (balanced for speed + robustness)
TOL_NO_DF = 3e-7    #tolerance for no friction
C_KEPLER  = 0.10   # Proportionality constant for step size 
TOL_DF    = 1e-6       # (b–d) per-unit-time tolerance (looser -> faster)
RTOL = 1e-10            # tolerance
FAC_MIN, FAC_MAX, SAFETY = 0.05, 3.0, 0.9 # some safety things to help rk4 efficiency
H_FLOOR = 1e-24        #Minimum for H

# -----------------------------
# Helpers
# -----------------------------
def min_advancing_step(t: float) -> float:
    return float(np.nextafter(t, np.inf) - t)

def v_circ(r: float) -> float:
    return math.sqrt(mu / r)

def vt_for_ellipse_at_apocenter(r_a, r_p):
    a = 0.5 * (r_a + r_p)
    return math.sqrt((mu / a) * (r_p / r_a))


# -----------------------------
# Dynamics
# -----------------------------
def a_df(vx, vy, A, B):
    v = math.hypot(vx, vy)
    denom = v**3 + B
    if denom <= 0.0: return 0.0, 0.0
    s = -A / denom
    return s*vx, s*vy

def rhs(t, y, A=0.0, B=1.0, use_df=False):
    x, y_, vx, vy = y
    r2 = x*x + y_*y_
    r  = math.sqrt(r2)
    invr3 = 1.0/(r2*r) if r>0 else 0.0
    ax = -mu * x * invr3
    ay = -mu * y_ * invr3
    if use_df:
        ax_d, ay_d = a_df(vx, vy, A, B)
        ax += ax_d; ay += ay_d
    return np.array([vx, vy, ax, ay], float)

def rk4_step(t, y, h, A, B, use_df):
    k1 = rhs(t,          y,            A, B, use_df)
    k2 = rhs(t+0.5*h,    y+0.5*h*k1,   A, B, use_df)
    k3 = rhs(t+0.5*h,    y+0.5*h*k2,   A, B, use_df)
    k4 = rhs(t+h,        y+h*k3,       A, B, use_df)
    return y + (h/6.0)*(k1+2*k2+2*k3+k4)

def rk4_try(t, y, h, tol, A, B, use_df):
    y_full  = rk4_step(t, y, h, A, B, use_df)
    y_half  = rk4_step(t, y, 0.5*h, A, B, use_df)
    y_half2 = rk4_step(t + 0.5*h, y_half, 0.5*h, A, B, use_df)

    err_vec = np.abs(y_half2 - y_full)
    scale   =  RTOL * np.maximum(np.abs(y_half2), np.abs(y_full))

    err_den  = float(tol * h + np.max(scale))
    if err_den == 0.0:
        err_norm = 0.0
    else:
        err_norm = float(np.max(err_vec)) / err_den

    accept = (err_norm <= 1.0)

    # step-size factor for order p=4 -> exponent 1/(p+1)=0.2
    if err_norm == 0.0:
        fac = FAC_MAX
    else:
        fac = SAFETY * (err_norm ** -0.2)
        fac = min(FAC_MAX, max(FAC_MIN, fac))

    return y_half2, accept, err_norm, fac


def integrate(y0, t0, h0, tol, A=0.0, B=1.0, use_df=False,
              r_stop=0.0, t_max=np.inf, max_steps=5_000_000,
              verbose_every=None):
    t = float(t0); y = np.array(y0, float); h = float(h0)
    ts=[t]; Ys=[y.copy()]
    for step in range(max_steps):
        r = math.hypot(y[0], y[1])
        if (r_stop>0 and r<=r_stop) or (t>=t_max): break
        # safe step limits
        adv_min = max(H_FLOOR, 4.0*min_advancing_step(t))
        if h < adv_min: h = adv_min
        h_cap = C_KEPLER * (max(r, r_s)**1.5) / math.sqrt(mu)
        h = min(h, h_cap)
        # trial
        y_new, ok, err, fac = rk4_try(t, y, h, tol, A, B, use_df)
        if verbose_every and step % verbose_every == 0:
            print(f"[{step:07d}] t={t:.6e} r={r:.3e} h={h:.3e} err={err:.3e} ok={ok}")
        if ok:
            t_next = t + h
            # guard against stagnation
            if t_next == t:
                h = max(h, 2.0 * adv_min)
                continue

            t = t_next
            y = y_new
            ts.append(t); Ys.append(y.copy())

            # never shrink h after an accepted step 
            if fac < 1.0:
                fac = 1.0
            h *= fac

        else:
            h *= fac
            if h < H_FLOOR: raise RuntimeError("Step-size underflow")
    return np.array(ts), np.vstack(Ys)


# (a) No DF, correct scaling and ≥10 orbits check

def part_a_no_df(orbits=10.5):
    r_a, r_p = 1.0, r_s
    v_ap = vt_for_ellipse_at_apocenter(r_a, r_p)   # ensures r_peri = r_s
    a = 0.5*(r_a + r_p)
    T = 2*math.pi * a**1.5 / math.sqrt(mu)        # Kepler period

    # integrate ≥ 10 orbits with no DF
    t, Y = integrate([r_a,0,0,v_ap], 0.0, 1e-4, TOL_NO_DF,
                     use_df=False, t_max=orbits*T, verbose_every=20000)

    # accuracy check: energy conservation 
    r = np.hypot(Y[:,0], Y[:,1])
    E = 0.5*(Y[:,2]**2 + Y[:,3]**2) - mu/r
    dE_abs  = E.max() - E.min()
    dE_rel  = dE_abs / max(1e-16, np.mean(np.abs(E)))

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Clean final version (no inset)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(Y[:, 0], Y[:, 1], lw=1.5, color="C0")
    ax.scatter([Y[0, 0]], [Y[0, 1]], s=25, color="C1", zorder=3, label="start")

    ax.set_xlim(-0.05, 1.02)
    ax.set_ylim(-8e-4, 8e-4)  # expanded slightly to frame the curve well
    ax.set_aspect("auto")     # stretch vertically so it’s readable
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Orbit without dynamical friction (≥10 orbits, r_peri = 1e-7)")
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "a_orbit_noDF.png"), dpi=220)
    plt.close(fig)

    # CSV + summary printout
    np.savetxt(os.path.join(OUT,"a_noDF_traj.csv"),
               np.column_stack([t, Y]), delimiter=",",
               header="t,x,y,vx,vy", comments="")
    print(f"[a] v_ap={v_ap:.6e}, T≈{T:.4f}, r_min={r.min():.3e}, r_max={r.max():.3e}")
    print(f"[a] Energy drift: ΔE={dE_abs:.3e}  (relative ≈ {dE_rel:.3e})")

# (b) DF with A=B=1; stop at r_s (give a reasonable t_max)
def part_b_df(A=1.0, B=1.0, v_mult=0.8, t_max=12.0):
    y0 = [1.0, 0.0, 0.0, v_mult*v_circ(1.0)]
    t, Y = integrate(y0, 0.0, 1e-2, TOL_DF, A=A, B=B, use_df=True,
                     r_stop=r_s, t_max=t_max, verbose_every=10000)
    r = np.hypot(Y[:,0], Y[:,1])

    # path
    plt.figure(figsize=(6,6))
    plt.plot(Y[:,0], Y[:,1], lw=1.0)
    # draw the r_s circle at the origin for reference
    th = np.linspace(0, 2*np.pi, 400)
    plt.plot(r_s*np.cos(th), r_s*np.sin(th), ls="--", lw=0.8)
    plt.axis("equal"); plt.xlabel("x"); plt.ylabel("y")
    plt.title(f"Orbit with dynamical friction (A=B=1)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"b_path_df.png"), dpi=160)

    # log r vs t
    plt.figure(figsize=(9,4))
    plt.plot(t, np.log(r), lw=1.0)
    plt.xlabel("t"); plt.ylabel("log r")
    plt.title("DF run (A=B=1): log r vs time")
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"b_logr_vs_t.png"), dpi=160)

    np.savetxt(os.path.join(OUT,"b_df_traj.csv"),
               np.column_stack([t, Y]), delimiter=",",
               header="t,x,y,vx,vy", comments="")
    hit = (r[-1] <= r_s + 1e-12)
    print(f"[b] finished t={t[-1]:.4f}, r_min={r.min():.3e}, hit_r_s={hit}")

# (c) Time to r_s vs ratio B/A (FAST)
def part_c_ratio_sweep(
        ratios=np.linspace(0.5, 10.0, 12),
        A_values=(0.5, 0.75, 1.0, 1.5, 2.0),   # ← five A’s
        preview_stop=1e-4,
        base_tmax=4.0,
        scale_tmax_by_ratio=True,
        cap_floor_units=6.0,                   # ~9 Myr (1 unit = 1.5 Myr)
        rerun_full_ratios=(0.5, 1.0, 2.0),
        t_max_full=15.0):

    UNIT_TO_MYR = 1.5

    rows = []
    # faster sweep knobs
    TOL_SWEEP = 5e-6
    C_KEPLER_SWEEP = 0.20
    global TOL_DF, C_KEPLER
    TOL_DF_old, C_KEPLER_old = TOL_DF, C_KEPLER
    TOL_DF, C_KEPLER = TOL_SWEEP, C_KEPLER_SWEEP

    for A in A_values:
        for R in ratios:
            B = R * A
            # base scaling with ratio
            t_cap = base_tmax * (R if scale_tmax_by_ratio else 1.0)
            # floor so small R doesn't timeout
            t_cap = max(t_cap, cap_floor_units)
            # NEW: scale cap by ~1/A (time ~ 1/A in the B >> v^3 regime)
            t_cap *= max(1.0, 1.0 / A)

            y0 = [1.0, 0.0, 0.0, 0.8 * v_circ(1.0)]
            t, Y = integrate(y0, 0.0, 1e-2, TOL_DF, A=A, B=B, use_df=True,
                             r_stop=preview_stop, t_max=t_cap)
            r_end = float(math.hypot(Y[-1, 0], Y[-1, 1]))
            hit = (r_end <= preview_stop * (1 + 1e-12))

            t_hit_Myr = t[-1] * UNIT_TO_MYR if hit else np.nan
            rows.append([A, R, B, t_hit_Myr, float(hit), t_cap * UNIT_TO_MYR])

    # restore
    TOL_DF, C_KEPLER = TOL_DF_old, C_KEPLER_old

    arr = np.array(rows, float)
    np.savetxt(os.path.join(OUT, "c_preview_table.csv"), arr, delimiter=",",
               header="A,B_over_A,B,time_to_r=1e-4_Myr_or_NaN,hit_flag,time_cap_Myr",
               comments="")

    # plot: one curve per A (label even if only misses)
    plt.figure(figsize=(8.6, 5.0))
    colors = plt.cm.tab10(np.linspace(0, 1, len(A_values)))
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]

    for idx, A in enumerate(A_values):
        sel = (arr[:, 0] == A)
        Rvals = arr[sel, 1]
        tvals = arr[sel, 3]
        hits  = (arr[sel, 4] == 1.0)
        caps  = arr[sel, 5]

        label_added = False
        if np.any(hits):
            plt.plot(Rvals[hits], tvals[hits],
                     marker=markers[idx % len(markers)], ms=5,
                     lw=1.6, color=colors[idx], label=f"A={A:g}")
            label_added = True

        miss = ~hits
        if np.any(miss):
            # draw open triangles for misses
            plt.plot(Rvals[miss], caps[miss],
                     marker=">", mfc="none", mec=colors[idx], mew=1.2,
                     ls="none", color=colors[idx],
                     label=(f"A={A:g}" if not label_added else None))

    plt.xlabel("B / A")
    plt.ylabel("Time to r = 1e-4 (Myr)")
    plt.title("Inspiral time vs B/A for multiple A (1 unit = 1.5 Myr)")
    plt.legend(title="A", frameon=False, ncol=min(3, len(A_values)))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "c_preview_t_vs_ratio.png"), dpi=170)

# (d) Sensitivity to initial speed at fixed ratio
def part_d_init_speed(
    ratio=1.0, A_fix=2.0,
    v_multipliers=(0.6, 0.7, 0.8, 0.9, 1.0),
    t_max=20.0,                  # longer cap to let most runs reach r_s
    preview_stop=None,           # e.g. 2e-4 for quick test; None = full r_s run
    tol=None,                    # optional tolerance override
    ckepler=None                 # optional C_KEPLER override
):
    """
    Runs the inspiral for several initial velocities to test sensitivity.
    """
    B_fix = ratio * A_fix
    target_r = r_s if preview_stop is None else float(preview_stop)

    # Local integrator settings
    TOL_LOCAL  = TOL_DF if tol is None else float(tol)
    CKEP_LOCAL = C_KEPLER if ckepler is None else float(ckepler)

    results = []
    for m in v_multipliers:
        y0 = [1.0, 0.0, 0.0, m * v_circ(1.0)]

        # temporarily override step-size cap by passing CKEP_LOCAL explicitly
        t, Y = integrate(
            y0, 0.0, 1e-2, TOL_LOCAL,
            A=A_fix, B=B_fix, use_df=True,
            r_stop=target_r, t_max=t_max, max_steps=2_000_000
        )

        r_end = math.hypot(Y[-1, 0], Y[-1, 1])
        hit = (r_end <= target_r * (1 + 1e-12))
        t_hit = 1.5*t[-1] if hit else float("nan")  #multiply time by 1.5 to get in Myr
        results.append([m, t_hit, float(hit)])

    res = np.array(results, float)  # cols: [v_mult, t_hit_or_nan, hit_flag]

    # Save results
    fname = "d_speed_sensitivity.csv" if preview_stop is None else "d_speed_preview.csv"
    header = (
        "v0_multiplier,time_to_r_s_or_NaN,hit_flag"
        if preview_stop is None
        else f"v0_multiplier,time_to_r_{target_r}_or_NaN,hit_flag"
    )
    np.savetxt(os.path.join(OUT, fname), res, delimiter=",", header=header, comments="")

    # Plot
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    mask_hit = res[:, 2] == 1.0
    if mask_hit.any():
        ax.plot(res[mask_hit, 0], res[mask_hit, 1], "o-", lw=1.3, label="reached target")
    if (~mask_hit).any():
        ax.plot(res[~mask_hit, 0], np.full((~mask_hit).sum(), t_max),
                "v", mfc="none", mec="gray", label="not reached (t_max)")

    target_label = "r_s" if preview_stop is None else f"r={preview_stop:g}"
    ax.set_xlabel("initial speed / v_circ")
    ax.set_ylabel(f"time to {target_label} (Myr)")
    ax.set_title(f"Sensitivity to initial speed (fixed B/A={ratio}, A={A_fix})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "d_speed_vs_time.png" if preview_stop is None else "d_speed_preview.png"), dpi=160)
    plt.close(fig)


# Run all parts
if __name__ == "__main__":
    print("\n=== (a) No DF ===")
    part_a_no_df(orbits=10.5)      # ≥ 10 orbits; auto-zoomed plot

    print("\n=== (b) DF, A=B=1 ===")
    part_b_df(A=1.0, B=1.0, v_mult=0.8, t_max=12.0)  # raise t_max if you want to guarantee r_s hit

    print("\n=== (c) Sweep ratios (fast preview) ===")
    part_c_ratio_sweep(
    ratios=np.linspace(0.5, 10.0, 10),
    A_values=(0.5, 0.75, 1.0, 1.5, 2.0),
    preview_stop=2e-4,     # slightly easier goal
    base_tmax=6.0,
    scale_tmax_by_ratio=True,
    rerun_full_ratios=(0.5, 1.0, 2.0),
    t_max_full=15.0
)

    print("\n=== (d) Initial-speed dependence ===")
    part_d_init_speed(preview_stop=None, t_max=20.0)
    print(f"\nAll outputs written to: {OUT}/")
