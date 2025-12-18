import numpy as np
import matplotlib.pyplot as plt

# =========================
# Dimer covering utilities
# =========================

# Encoding for each site:
# 0 empty
# 1 paired to right   (partner at (i, j+1))
# 2 paired to left    (partner at (i, j-1))
# 3 paired to up      (partner at (i-1, j))
# 4 paired to down    (partner at (i+1, j))

DIRS = {
    "R": (0, +1, 1, 2),
    "L": (0, -1, 2, 1),
    "U": (-1, 0, 3, 4),
    "D": (+1, 0, 4, 3),
}

def in_bounds(i, j, L):
    return (0 <= i < L) and (0 <= j < L)

def are_connected(state, a, b):
    """Return True if sites a and b are endpoints of the same dimer."""
    (i1, j1), (i2, j2) = a, b
    if state[i1, j1] == 0 or state[i2, j2] == 0:
        return False

    # Check if a points to b with correct direction code
    di = i2 - i1
    dj = j2 - j1
    if di == 0 and dj == 1:
        return state[i1, j1] == 1 and state[i2, j2] == 2
    if di == 0 and dj == -1:
        return state[i1, j1] == 2 and state[i2, j2] == 1
    if di == -1 and dj == 0:
        return state[i1, j1] == 3 and state[i2, j2] == 4
    if di == 1 and dj == 0:
        return state[i1, j1] == 4 and state[i2, j2] == 3
    return False

def add_dimer(state, a, b):
    """Assumes both empty and adjacent; writes orientation codes."""
    (i1, j1), (i2, j2) = a, b
    di = i2 - i1
    dj = j2 - j1
    if di == 0 and dj == 1:
        state[i1, j1] = 1; state[i2, j2] = 2
    elif di == 0 and dj == -1:
        state[i1, j1] = 2; state[i2, j2] = 1
    elif di == -1 and dj == 0:
        state[i1, j1] = 3; state[i2, j2] = 4
    elif di == 1 and dj == 0:
        state[i1, j1] = 4; state[i2, j2] = 3
    else:
        raise ValueError("add_dimer called on non-adjacent pair")

def remove_dimer(state, a, b):
    """Assumes a and b are connected by a dimer; clears them."""
    (i1, j1), (i2, j2) = a, b
    state[i1, j1] = 0
    state[i2, j2] = 0

def count_dimers(state):
    """Each dimer occupies two sites. Count dimers = (# occupied sites)/2."""
    return int(np.count_nonzero(state) // 2)

def sample_random_adjacent_pair(L, rng):
    """Choose an undirected nearest-neighbor bond uniformly on an open LxL lattice."""
    # Choose a random site, then choose a direction that stays in bounds.
    i = rng.integers(0, L)
    j = rng.integers(0, L)
    # Possible neighbors
    neighbors = []
    if i > 0: neighbors.append((i-1, j))
    if i < L-1: neighbors.append((i+1, j))
    if j > 0: neighbors.append((i, j-1))
    if j < L-1: neighbors.append((i, j+1))
    ni, nj = neighbors[rng.integers(0, len(neighbors))]
    return (i, j), (ni, nj)

def temperature_exponential(t, T0, tau, Tmin=1e-6):
    T = T0 * np.exp(-t / tau)
    return max(T, Tmin)

def anneal_dimers(L=50, nsteps=200_000, T0=5.0, tau=10_000, seed=0,
                 snapshot_steps=(1000, None), record_trace=False):
    """
    Simulated annealing for the dimer covering problem.

    Moves (Newman):
      i) choose adjacent sites at random
     ii) if they contain a dimer connecting them: remove with prob exp(-1/T)
    iii) if both empty: always add a dimer (energy decreases)
     iv) otherwise: do nothing

    Energy: E = -N_dimer.
    Removing one dimer increases E by +1 -> acceptance exp(-1/T).
    Adding decreases E by -1 -> always accept.
    """
    rng = np.random.default_rng(seed)
    state = np.zeros((L, L), dtype=np.int8)

    early_state = None
    final_state = None

    trace = [] if record_trace else None

    # snapshot_steps: (early_step, final_step) with final_step=None => end
    early_step = snapshot_steps[0]
    final_step = snapshot_steps[1]

    for t in range(nsteps):
        T = temperature_exponential(t, T0=T0, tau=tau)

        a, b = sample_random_adjacent_pair(L, rng)

        sa = state[a]
        sb = state[b]

        if sa == 0 and sb == 0:
            # add dimer always
            add_dimer(state, a, b)
        elif are_connected(state, a, b):
            # propose removal, accept with exp(-1/T)
            if rng.random() < np.exp(-1.0 / T):
                remove_dimer(state, a, b)
        else:
            # do nothing
            pass

        if record_trace and (t % 100 == 0):
            trace.append(count_dimers(state))

        if t == early_step:
            early_state = state.copy()

        if final_step is not None and t == final_step:
            final_state = state.copy()

    if early_state is None:
        early_state = state.copy()
    if final_state is None:
        final_state = state.copy()

    return state, early_state, final_state, trace

def plot_configuration(state, title, outname):
    """
    Plot dimers as line segments on a grid.
    """
    L = state.shape[0]
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(-0.5, L - 0.5)

    # draw dimers once: only draw from "right" and "down" codes to avoid doubles
    for i in range(L):
        for j in range(L):
            code = state[i, j]
            if code == 1:  # right
                ax.plot([j, j+1], [L-1-i, L-1-i], linewidth=2)
            elif code == 4:  # down (in array coords), visually it's -y direction
                ax.plot([j, j], [L-1-i, L-2-i], linewidth=2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outname, dpi=200)
    plt.close()

def plot_early_vs_final(early, final, tau, outname):
    L = early.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, st, ttl in zip(axes, [early, final], ["Early state", "Final state"]):
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, L - 0.5)
        ax.set_ylim(-0.5, L - 0.5)
        for i in range(L):
            for j in range(L):
                code = st[i, j]
                if code == 1:  # right
                    ax.plot([j, j+1], [L-1-i, L-1-i], linewidth=2)
                elif code == 4:  # down
                    ax.plot([j, j], [L-1-i, L-2-i], linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(ttl)
    fig.suptitle(f"Dimer annealing on {L}x{L} lattice (tau = {tau})")
    plt.tight_layout()
    plt.savefig(outname, dpi=200)
    plt.close()

def write_latex_table(rows, outname_tex="dimer_tau_table.tex"):
    """
    Write a LaTeX tabular environment you can \\input{} directly.
    rows: list of dicts with keys: tau, steps, Nd, frac
    """
    with open(outname_tex, "w") as f:
        f.write("\\begin{tabular}{rrrr}\n")
        f.write("\\hline\n")
        f.write("$\\tau$ & steps & $N_{\\rm dimer}$ & coverage $f=2N/L^2$\\\\\n")
        f.write("\\hline\n")
        for r in rows:
            f.write(f"{r['tau']} & {r['steps']} & {r['Nd']} & {r['frac']:.4f}\\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

def main():
    L = 50

    # === (a) + (b): compare cooling schedules ===
    # You can tweak these; these are a sensible spread around Newman's suggestion tau=10000.
    taus = [2000, 5000, 10000, 20000]
    nsteps = 300_000
    T0 = 5.0

    results = []

    # Pick one "representative" tau for the early-vs-final plot (Newman suggests 10000)
    tau_show = 10000

    for k, tau in enumerate(taus):
        final_state, early_state, late_state, _ = anneal_dimers(
            L=L, nsteps=nsteps, T0=T0, tau=tau, seed=123 + k,
            snapshot_steps=(2000, None), record_trace=False
        )

        Nd = count_dimers(final_state)
        frac = (2.0 * Nd) / (L * L)

        results.append({"tau": tau, "steps": nsteps, "Nd": Nd, "frac": frac})

        if tau == tau_show:
            plot_early_vs_final(
                early_state, late_state, tau=tau,
                outname="dimer_early_vs_final_tau10000.png"
            )
            # also save final configuration alone if you want
            plot_configuration(final_state,
                               title=f"Final configuration (tau={tau})",
                               outname="dimer_final_tau10000.png")

    # Write LaTeX-ready table
    write_latex_table(results, outname_tex="dimer_tau_table.tex")

    # Also print to terminal
    print("Cooling schedule comparison:")
    for r in results:
        print(f"tau={r['tau']:>5}  Nd={r['Nd']:>4}  coverage f={r['frac']:.4f}")

    print("\nWrote:")
    print("  dimer_early_vs_final_tau10000.png")
    print("  dimer_final_tau10000.png")
    print("  dimer_tau_table.tex")

if __name__ == "__main__":
    main()
