import numpy as np
import matplotlib.pyplot as plt

def total_energy(spins: np.ndarray, J: float = 1.0) -> float:
    """
    Total energy with periodic BCs, counting each bond once:
    E = -J sum_{i,j} [ s_{i,j} s_{i+1,j} + s_{i,j} s_{i,j+1} ].
    """
    right = np.roll(spins, shift=-1, axis=1)
    up    = np.roll(spins, shift=-1, axis=0)
    return -J * np.sum(spins * right + spins * up)

def metropolis_step(spins: np.ndarray, T: float, J: float, rng: np.random.Generator) -> int:
    """
    Perform one Metropolis update (one attempted spin flip).
    Returns deltaM (change in total magnetization) which is +/-2 or 0.
    """
    L = spins.shape[0]
    i = rng.integers(0, L)
    j = rng.integers(0, L)

    s = spins[i, j]
    # nearest neighbors with periodic boundaries
    nn_sum = (
        spins[(i + 1) % L, j] +
        spins[(i - 1) % L, j] +
        spins[i, (j + 1) % L] +
        spins[i, (j - 1) % L]
    )

    dE = 2.0 * J * s * nn_sum  # flipping changes energy locally
    if dE <= 0.0 or rng.random() < np.exp(-dE / T):
        spins[i, j] = -s
        return -2 * s  # M_new - M_old = (-s - s) = -2s
    return 0

def run_ising(
    L: int = 20,
    T: float = 1.0,
    J: float = 1.0,
    nsteps: int = 1_000_000,
    seed: int = 0,
    snapshot_times=None,
):
    rng = np.random.default_rng(seed)
    spins = rng.choice([-1, 1], size=(L, L))
    M = int(np.sum(spins))

    Ms = np.empty(nsteps, dtype=np.int32)
    snapshots = {}

    snapshot_set = set(snapshot_times) if snapshot_times is not None else set()

    for t in range(nsteps):
        dM = metropolis_step(spins, T=T, J=J, rng=rng)
        M += dM
        Ms[t] = M
        if t in snapshot_set:
            snapshots[t] = spins.copy()

    E = total_energy(spins, J=J)
    return spins, Ms, E, snapshots

def log_spaced_times(nsteps: int, nsnaps: int) -> np.ndarray:
    """
    Logarithmically spaced integer times in [0, nsteps-1].
    Includes t=0 and t=nsteps-1.
    """
    if nsnaps < 2:
        return np.array([0], dtype=int)
    # use 1..nsteps to avoid log(0), then shift back
    vals = np.logspace(0, np.log10(nsteps), nsnaps)
    ts = np.unique(np.clip(vals.astype(int) - 1, 0, nsteps - 1))
    if ts[0] != 0:
        ts = np.insert(ts, 0, 0)
    if ts[-1] != nsteps - 1:
        ts = np.append(ts, nsteps - 1)
    return ts

def plot_magnetization(Ms: np.ndarray, L: int, T: float, outname: str):
    steps = np.arange(len(Ms))
    plt.figure()
    plt.plot(steps, Ms)
    plt.xlabel("Monte Carlo step")
    plt.ylabel(r"Magnetization $M=\sum s_{ij}$")
    plt.title(f"2D Ising Metropolis: L={L}, T={T}")
    plt.tight_layout()
    plt.savefig(outname, dpi=200)
    plt.close()

def plot_snapshots(snapshots: dict, L: int, T: float, outname: str, max_panels: int = 12):
    """
    Plot up to max_panels snapshots sorted by time.
    """
    times = sorted(snapshots.keys())
    if len(times) > max_panels:
        # downselect evenly in index-space
        idx = np.linspace(0, len(times) - 1, max_panels).astype(int)
        times = [times[i] for i in idx]

    n = len(times)
    ncols = 4 if n >= 4 else n
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax in axes[n:]:
        ax.axis("off")

    for k, t in enumerate(times):
        ax = axes[k]
        ax.imshow(snapshots[t], interpolation="nearest")
        ax.set_title(f"t = {t}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Spin configurations (log-spaced times), L={L}, T={T}")
    plt.tight_layout()
    plt.savefig(outname, dpi=200)
    plt.close()

def main():
    L = 20
    J = 1.0
    nsteps = 1_000_000

    # (c) Long magnetization run at T=1
    T = 1.0
    times = log_spaced_times(nsteps, nsnaps=12)
    spins, Ms, E, snaps = run_ising(L=L, T=T, J=J, nsteps=nsteps, seed=1, snapshot_times=times)

    plot_magnetization(Ms, L=L, T=T, outname="ising_M_vs_time_T1.png")
    plot_snapshots(snaps, L=L, T=T, outname="ising_snapshots_T1.png")

    print("=== T = 1 run ===")
    print(f"Final energy E = {E:.1f}")
    print(f"Final magnetization M = {Ms[-1]} (out of {L*L})")

    # (e) Shorter snapshot runs at T=2 and T=3 (still log-spaced)
    for T, seed in [(2.0, 2), (3.0, 3)]:
        nsteps_short = 200_000
        times_short = log_spaced_times(nsteps_short, nsnaps=12)
        spins, Ms, E, snaps = run_ising(
            L=L, T=T, J=J, nsteps=nsteps_short, seed=seed, snapshot_times=times_short
        )
        plot_magnetization(Ms, L=L, T=T, outname=f"ising_M_vs_time_T{int(T)}.png")
        plot_snapshots(snaps, L=L, T=T, outname=f"ising_snapshots_T{int(T)}.png")
        print(f"=== T = {T} run ===")
        print(f"Final energy E = {E:.1f}")
        print(f"Final magnetization M = {Ms[-1]} (out of {L*L})")

    print("\nSaved figures:")
    print("  ising_M_vs_time_T1.png")
    print("  ising_snapshots_T1.png")
    print("  ising_M_vs_time_T2.png")
    print("  ising_snapshots_T2.png")
    print("  ising_M_vs_time_T3.png")
    print("  ising_snapshots_T3.png")

if __name__ == "__main__":
    main()
