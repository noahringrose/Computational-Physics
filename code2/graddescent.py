import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Test Function Example: Quadratic Bowl
def test_func(x):
    """Simple test function: f(x, y) = (x-1)^2 + 2*(y+2)^2"""
    return (x[0] - 2)**2 + (x[1] - 2)**2

# Define numerical gradient using central difference
def numerical_gradient(func, params, eps=1e-6):
    grad = np.zeros_like(params)
    for i in range(len(params)):
        dp = np.zeros_like(params)
        dp[i] = eps
        grad[i] = (func(params + dp) - func(params - dp)) / (2 * eps)
    return grad

#Define gradient descent algorithm for general function, initial parameters, bounds, epsilon threshold, initial learning rate,
#max iterations, hrinking factor for backtracking, minimum learning rate, and minimum change in parameters

def gradient_descent(func, init_params, bounds=None,
                     grad_eps=1e-6, init_lr=0.5, beta=0.5,
                     min_lr=1e-8, maxiter=5000, tol=1e-9):
    #initialize parameters
    params = np.array(init_params, dtype=float)
    #reference value for function
    fval = func(params)
    #history of descent
    history = {'params': [], 'fval': []}
    for k in range(maxiter):
        history['params'].append(params.copy())
        history['fval'].append(float(fval))
        grad = numerical_gradient(func, params, eps=grad_eps)
        lr = init_lr
        success = False
        while lr >= min_lr:
            # descend
            candidate = params - lr * grad
            if bounds is not None:
                for i in range(len(candidate)):
                    candidate[i] = np.clip(candidate[i], bounds[i][0], bounds[i][1])
            f_candidate = func(candidate)
            #check if function decreased after descent
            if f_candidate < fval:
                params = candidate
                fval = f_candidate
                success = True
                break
            # if not, scale learning rate down and try again
            lr *= beta
        if not success:
            break
        #check if descent is smaller than our tolerance and stop if so
        if np.linalg.norm(lr * grad) < tol:
            break
    # return the values of parameters which minimize function as well as the steps we took to get there    
    return params, history

# Run test function optimization
init_guess = np.array([3.0, -5.0])
best_test, hist_test = gradient_descent(test_func, init_guess)
print("=== Test Function Optimization ===")
print(f"Start: {init_guess}, Found minimum: {best_test}")
print(f"Final function value: {hist_test['fval'][-1]:.6e}")

# Plot the convergence to minimum value
plt.figure(figsize=(5, 4))
plt.semilogy(hist_test['fval'], '-o')
plt.xlabel("Iteration")
plt.ylabel("f(x)")
plt.title("Test Function Convergence")
plt.tight_layout()
plt.savefig("test_function_convergence.png", dpi=150)
plt.close()


# COSMOS SMF Data Fit using Schechter Function

# Load data: columns = log10(M_gal), n(M_gal), error(n)
data = np.loadtxt("code2/smf_cosmos.dat")
logM = data[:, 0]
n_obs = data[:, 1]
sigma = data[:, 2]

# Schechter function per dex
def schechter_per_dex(logM, phi_star, M_star, alpha):
    M = 10.0 ** logM
    x = M / M_star
    return phi_star * x ** (alpha + 1) * np.exp(-x) * np.log(10.0)

# Chi-squared function
def chi2(params):
    log10_phi, log10_Mstar, alpha = params
    phi = 10.0 ** log10_phi
    Mstar = 10.0 ** log10_Mstar
    model = schechter_per_dex(logM, phi, Mstar, alpha)
    return np.sum(((n_obs - model) ** 2) / (sigma ** 2))

# Fit Schechter function
bounds = np.array([[-8.0, -1.0], [8.0, 12.0], [-3.0, 1.0]])
starts = [
    [-3.0, 10.0, -1.2],
    [-2.5, 10.5, -1.0],
    [-4.0, 9.5, -1.5],
    [-3.5, 11.0, -0.8]
]

results = []
for init in starts:
    found, history = gradient_descent(chi2, init, bounds=bounds)
    results.append({'init': init, 'found': found, 'chi2': history['fval'][-1], 'history': history})

best = min(results, key=lambda r: r['chi2'])
best_p = best['found']
log10_phi, log10_Mstar, alpha = best_p
phi_star = 10 ** log10_phi
M_star = 10 ** log10_Mstar

print("\n=== Schechter Fit Results ===")
print(f"log10(phi*) = {log10_phi:.6f}")
print(f"log10(M*)   = {log10_Mstar:.6f}")
print(f"alpha       = {alpha:.6f}")
print(f"phi* = {phi_star:.4e},  M* = {M_star:.4e}")
print(f"Final chi^2 = {best['chi2']:.6f}")

# Save results and plots
outdir = Path("schechter_fit_results")
outdir.mkdir(exist_ok=True)

# Chi2 vs iteration
plt.figure(figsize=(6,4))
plt.semilogy(best['history']['fval'], '-o')
plt.xlabel("Iteration")
plt.ylabel(r"$\chi^2$")
plt.title(r"Chi$^2$ vs Iteration")
plt.tight_layout()
plt.savefig(outdir / "chi2_vs_iter.png", dpi=150)
plt.close()

# Best-fit Schechter vs data
plt.figure(figsize=(6,4))
plt.errorbar(logM, n_obs, yerr=sigma, fmt='o', label='Data')
model = schechter_per_dex(logM, phi_star, M_star, alpha)
plt.plot(logM, model, label='Best-fit Schechter', color='C1')
plt.yscale('log')
plt.xlabel(r'$\log_{10} M_{\mathrm{gal}}$')
plt.ylabel(r'$n(M_{\mathrm{gal}})$ [1/dex/Volume]')
plt.legend()
plt.title('Schechter Fit to COSMOS Data')
plt.tight_layout()
plt.savefig(outdir / "schechter_fit_loglog.png", dpi=150)
plt.close()

# Save results to text file
with open(outdir / "fit_results.txt", "w") as f:
    f.write("Best-fit Schechter parameters:\n")
    f.write(f"log10(phi*) = {log10_phi:.6f}\n")
    f.write(f"log10(M*)   = {log10_Mstar:.6f}\n")
    f.write(f"alpha       = {alpha:.6f}\n")
    f.write(f"phi* = {phi_star:.4e}\n")
    f.write(f"M*  = {M_star:.4e}\n")
    f.write(f"chi^2 = {best['chi2']:.6f}\n")

print(f"Plots and results saved to: {outdir.resolve()}")
