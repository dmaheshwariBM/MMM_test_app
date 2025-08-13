import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

def _adj_r2(r2, n, p):
    return 1 - (1-r2)*(n-1)/(max(n-p-1,1))

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    X_ = X.assign(_const=1.0)
    vifs = []
    for i, c in enumerate(X.columns):
        vifs.append((c, float(variance_inflation_factor(X_.values, i))))
    return pd.DataFrame(vifs, columns=["feature","VIF"])

def ols_model(X: pd.DataFrame, y: pd.Series):
    model = LinearRegression()
    model.fit(X, y)
    yhat = model.predict(X)
    r2 = r2_score(y, yhat)
    return model, {
        "R2": float(r2),
        "AdjR2": float(_adj_r2(r2, len(y), X.shape[1])),
        "RMSE": float(mean_squared_error(y, yhat, squared=False)),
        "MAE": float(mean_absolute_error(y, yhat))
    }, yhat

def ridge_model_cv(X: pd.DataFrame, y: pd.Series, alphas=None, cv=5):
    if alphas is None:
        alphas = np.logspace(-3,3,50)
    model = RidgeCV(alphas=alphas, cv=cv, scoring=None)
    model.fit(X, y)
    yhat = model.predict(X)
    r2 = r2_score(y, yhat)
    return model, {
        "R2": float(r2),
        "AdjR2": float(_adj_r2(r2, len(y), X.shape[1])),
        "RMSE": float(mean_squared_error(y, yhat, squared=False)),
        "MAE": float(mean_absolute_error(y, yhat)),
        "BestAlpha": float(model.alpha_)
    }, yhat

def lasso_model_cv(X: pd.DataFrame, y: pd.Series, alphas=None, cv=5):
    if alphas is None:
        alphas = np.logspace(-3,3,50)
    model = LassoCV(alphas=alphas, cv=cv, random_state=0, max_iter=10000)
    model.fit(X, y)
    yhat = model.predict(X)
    r2 = r2_score(y, yhat)
    return model, {
        "R2": float(r2),
        "AdjR2": float(_adj_r2(r2, len(y), X.shape[1])),
        "RMSE": float(mean_squared_error(y, yhat, squared=False)),
        "MAE": float(mean_absolute_error(y, yhat)),
        "BestAlpha": float(model.alpha_)
    }, yhat

def contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    coefs = model.coef_ if hasattr(model, "coef_") else np.zeros(X.shape[1])
    impact = X.values * coefs
    contrib = impact.mean(axis=0)
    share = contrib / (contrib.sum() + 1e-12)
    return pd.DataFrame({"feature": X.columns, "coef": coefs, "avg_contribution": contrib, "share": share})
