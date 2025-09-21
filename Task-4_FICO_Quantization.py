# Task 4 – FICO Score Bucketing
# ------------------------------------------------------------------------------
# We partition FICO scores into K buckets using dynamic programming.
# Two objectives are supported:
#   (1) MSE  – minimize within-bucket squared error of FICO values
#   (2) LL   – maximize binomial log-likelihood of defaults in each bucket (with smoothing)
#
# Each bucket is summarized by count, defaults, and empirical PD.
# Ratings are assigned so that Rating=1 = best (highest FICO), Rating=K = worst.
# The output includes:
#   - bucket boundaries
#   - summary table with stats
#   - a mapping function map_fico_to_rating


import numpy as np
import pandas as pd
from typing import Literal, Dict, Any, Tuple, List


# unique FICO values

def _prep_fico_aggregates(df: pd.DataFrame, fico_col: str, target_col: str):
    """
    Collapse duplicate FICO scores into unique rows with weights:
      n = number of observations at that FICO
      k = number of defaults at that FICO
    Returns arrays x (unique FICO), n (counts), k (defaults), all sorted by FICO.
    """
    g = (df[[fico_col, target_col]]
         .dropna()
         .groupby(fico_col, as_index=False)
         .agg(n=(target_col, "size"), k=(target_col, "sum"))
         .sort_values(fico_col)
         .reset_index(drop=True))
    x = g[fico_col].to_numpy(dtype=float)
    n = g["n"].to_numpy(dtype=float)
    k = g["k"].to_numpy(dtype=float)
    return x, n, k, g


# Weighted prefix sums

def _prefix_sums_weighted(x: np.ndarray, n: np.ndarray, k: np.ndarray):
    """
    Build weighted prefix sums to query segment stats in O(1).
    """
    one = np.ones_like(n)
    ps = {
        "cnt":   np.concatenate([[0.0], np.cumsum(n)]),
        "sumx":  np.concatenate([[0.0], np.cumsum(n * x)]),
        "sumxx": np.concatenate([[0.0], np.cumsum(n * x * x)]),
        "sumk":  np.concatenate([[0.0], np.cumsum(k)]),
    }
    return ps

def _seg_wcounts(ps, i: int, j: int):
    """
    Return weighted stats on segment [i:j).
    Nw = sum of counts, Sx = sum(n*x), Sxx = sum(n*x^2), Kd = sum of defaults.
    """
    Nw  = ps["cnt"][j]   - ps["cnt"][i]
    Sx  = ps["sumx"][j]  - ps["sumx"][i]
    Sxx = ps["sumxx"][j] - ps["sumxx"][i]
    Kd  = ps["sumk"][j]  - ps["sumk"][i]
    return Nw, Sx, Sxx, Kd

# Segment objectives

def _mse_cost_w(ps, i: int, j: int) -> float:
    """
    Weighted SSE within segment [i:j):
      SSE = sum(n*x^2) - (sum(n*x))^2 / sum(n)
    """
    Nw, Sx, Sxx, _ = _seg_wcounts(ps, i, j)
    if Nw <= 0:
        return 0.0
    return float(Sxx - (Sx * Sx) / Nw)

def _ll_score_w(ps, i: int, j: int, alpha: float = 1e-6) -> float:
    """
    Binomial log-likelihood on [i:j), with Laplace/Jeffreys smoothing to avoid log(0):
      p = (k + alpha) / (n + 2*alpha)
      LL = k*log(p) + (n-k)*log(1-p)
    """
    Nw, _, __, Kd = _seg_wcounts(ps, i, j)
    if Nw <= 0:
        return 0.0
    p = (Kd + alpha) / (Nw + 2 * alpha)
    return float(Kd * np.log(p) + (Nw - Kd) * np.log(1 - p))

# Weighted DP (O(K * U^2))

def _optimal_buckets_weighted(
    x: np.ndarray,
    n: np.ndarray,
    k: np.ndarray,
    K: int,
    objective: Literal["mse", "ll"] = "ll",
    alpha: float = 1e-6,
    min_bucket_size: float = 100.0,
) -> Tuple[List[int], float]:
    """
    Return cut indices (monotone, ending at U) that optimize the objective over unique FICO x.
    For "mse" we minimize total SSE; for "ll" we maximize total LL.
    min_bucket_size is enforced in 'number of observations' units.
    """
    U = len(x)
    if U == 0 or K <= 0:
        return [0], 0.0
    K = min(K, U)

    ps = _prefix_sums_weighted(x, n, k)

    if objective == "mse":
        dp   = np.full((K + 1, U + 1), np.inf)
        back = np.full((K + 1, U + 1), -1, dtype=int)
        dp[0, 0] = 0.0

        for kb in range(1, K + 1):
            for j in range(1, U + 1):
                best, best_i = np.inf, -1
                # enforce min bucket size by skipping i that makes segment too small
                i_low = 0
                if min_bucket_size > 0:
                    while i_low < j and (ps["cnt"][j] - ps["cnt"][i_low]) < min_bucket_size:
                        i_low += 1
                start_i = max(kb - 1, i_low)
                for i in range(start_i, j):
                    cost = dp[kb - 1, i] + _mse_cost_w(ps, i, j)
                    if cost < best:
                        best, best_i = cost, i
                dp[kb, j], back[kb, j] = best, best_i

        cuts = [U]; j = U
        for kb in range(K, 0, -1):
            i = back[kb, j]; cuts.append(i); j = i
        return sorted(cuts), float(dp[K, U])

    else:
        dp   = np.full((K + 1, U + 1), -np.inf)
        back = np.full((K + 1, U + 1), -1, dtype=int)
        dp[0, 0] = 0.0

        for kb in range(1, K + 1):
            for j in range(1, U + 1):
                best, best_i = -np.inf, -1
                i_low = 0
                if min_bucket_size > 0:
                    while i_low < j and (ps["cnt"][j] - ps["cnt"][i_low]) < min_bucket_size:
                        i_low += 1
                start_i = max(kb - 1, i_low)
                for i in range(start_i, j):
                    score = dp[kb - 1, i] + _ll_score_w(ps, i, j, alpha=alpha)
                    if score > best:
                        best, best_i = score, i
                dp[kb, j], back[kb, j] = best, best_i

        cuts = [U]; j = U
        for kb in range(K, 0, -1):
            i = back[kb, j]; cuts.append(i); j = i
        return sorted(cuts), float(dp[K, U])

# Public API

def fit_fico_rating_map(
    df: pd.DataFrame,
    fico_col: str = "fico_score",
    target_col: str = "default",
    n_buckets: int = 10,
    objective: Literal["mse", "ll"] = "ll",
    smoothing: float = 1e-6,
    min_bucket_size: float = 100.0,
) -> Dict[str, Any]:
    """
    Fit an optimal quantization of FICO into n_buckets using weighted DP.
    Returns:
      - 'objective': used objective
      - 'score': objective value
      - 'boundaries': list of max FICO per bucket (ascending)
      - 'table': DataFrame with ranges, counts, defaults, PD, rating
      - 'map_func': callable map_fico_to_rating(fico) -> rating (1=best ... K=worst)
    """
    # Aggregate by unique FICO
    x, n, k, agg_df = _prep_fico_aggregates(df, fico_col=fico_col, target_col=target_col)

    if len(x) == 0:
        raise ValueError("No valid rows after dropping NA on fico/default.")

    # Solve DP over unique scores
    cuts, obj = _optimal_buckets_weighted(
        x, n, k,
        K=n_buckets,
        objective=objective,
        alpha=smoothing,
        min_bucket_size=min_bucket_size
    )

    # Build summary table from cuts
    rows = []
    for b in range(1, len(cuts)):
        i0, i1 = cuts[b - 1], cuts[b]
        seg = agg_df.iloc[i0:i1]
        N = float(seg["n"].sum())
        Kd = float(seg["k"].sum())
        pd_hat = Kd / N if N > 0 else np.nan
        lo = float(seg[fico_col].min()) if len(seg) else np.nan
        hi = float(seg[fico_col].max()) if len(seg) else np.nan
        rows.append({
            "bucket_idx": b - 1,
            "fico_min": lo,
            "fico_max": hi,
            "n": int(N),
            "k": int(Kd),
            "pd_hat": pd_hat
        })
    summary = pd.DataFrame(rows)

    # Assign ratings: higher FICO → better
    # Our cuts are ascending, so bucket_idx=0 is lowest FICO.
    summary["rating"] = n_buckets - summary["bucket_idx"]

    # Boundaries = max FICO per bucket, ascending
    boundaries = summary.sort_values("bucket_idx")["fico_max"].astype(float).tolist()

    bounds = summary[["fico_max", "rating"]].sort_values("fico_max").to_numpy()

    def map_fico_to_rating(fico: float) -> int:
        for bmax, r in bounds:
            if fico <= bmax:
                return int(r)
        
        return int(summary["rating"].min())

    return {
        "objective": objective,
        "score": obj,
        "boundaries": boundaries,
        "table": summary.sort_values("rating"),
        "map_func": map_fico_to_rating,
    }

# Example usage

df = pd.read_csv("Task 3 and 4_Loan_Data.csv")
result = fit_fico_rating_map(
    df,
    fico_col="fico_score",
    target_col="default",
    n_buckets=10,
    objective="ll",
    smoothing=1e-6,
    min_bucket_size=200.0
)
print("Objective:", result["objective"], "Score:", result["score"])
print(result["table"])
print("Boundaries:", result["boundaries"])
print("Rating(FICO=720):", result)
