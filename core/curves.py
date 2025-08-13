import pandas as pd
import numpy as np

def linear_curve(spend_series: pd.Series, coef: float, intercept: float = 0.0):
    max_spend = max(1.0, float(spend_series.max()))
    spends = np.linspace(0, max_spend*1.5, 60)
    impact = intercept + coef*spends
    return pd.DataFrame({"Spend": spends, "Impact": impact})

def hill_curve(spend_series: pd.Series, vmax: float, k: float, theta: float):
    max_spend = max(1.0, float(spend_series.max()))
    spends = np.linspace(0, max_spend*1.5, 60)
    resp = vmax * (spends**k)/(spends**k + theta**k + 1e-12)
    return pd.DataFrame({"Spend": spends, "Impact": resp})
