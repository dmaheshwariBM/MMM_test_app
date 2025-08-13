import numpy as np
from scipy.optimize import minimize

def optimize_budget_hill(channels, vmax, k, theta, total_budget, minmax):
    # channels: list[str]; params dicts keyed by channel
    idx = {c:i for i,c in enumerate(channels)}
    def neg_obj(x):
        val = 0.0
        for c in channels:
            i = idx[c]
            s = max(0.0, x[i])
            val += vmax[c] * (s**k[c])/(s**k[c] + theta[c]**k[c] + 1e-9)
        return -val

    x0 = np.array([ (minmax[c][0]+minmax[c][1])/2.0 for c in channels ])
    bnds = [ minmax[c] for c in channels ]
    cons = [{"type":"eq","fun": lambda x: np.sum(x) - total_budget}]
    res = minimize(neg_obj, x0, bounds=bnds, constraints=cons, method="SLSQP")
    return res.x, -res.fun, res
