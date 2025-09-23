import numpy as np
import matplotlib.pyplot as plt

def forward_diff(f, x, h):
    x, h = np.float32(x), np.float32(h)
    return (f(x + h) - f(x)) / h

def central_diff(f, x, h):
    x, h = np.float32(x), np.float32(h)
    return (f(x + h) - f(x - h)) / (2*h)

def extrapolated_diff(f, x, h):
    x, h = np.float32(x), np.float32(h)
    f1 = (f(x + h) - f(x - h)) / (2*h)
    f2 = (f(x + 2*h) - f(x - 2*h)) / (4*h)
    return (4/3) * f1 - (1/3) * f2

def f_cos(x): return np.float32(np.cos(x))
def f_exp(x): return np.float32(np.exp(x))

def d_cos(x): return -np.sin(x)
def d_exp(x): return np.exp(x)

functions = [(f_cos, d_cos, "cos(x)"), (f_exp, d_exp, "exp(x)")]
x_values = [0.1, 10.0]

methods = {
    "Forward": forward_diff,
    "Central": central_diff,
    "Extrapolated": extrapolated_diff
}

hs = np.logspace(-1, -10, 50, dtype=np.float32)

for f, df_exact, fname in functions:
    for x in x_values:
        true_val = df_exact(x)
        plt.figure(figsize=(6,4))
        for mname, method in methods.items():
            errors = []
            for h in hs:
                approx = method(f, x, h)
                rel_err = abs((approx - true_val) / true_val)
                errors.append(rel_err)
            plt.loglog(hs, errors, label=mname)
        plt.xlabel("Step size h")
        plt.ylabel("Relative error ε")
        plt.title(f"Derivative of {fname} at x={x}")
        plt.legend()
        plt.grid(True, which="both")
        plt.savefig(f"error_{fname.replace('(','').replace(')','')}_x{x}.png", dpi=150)