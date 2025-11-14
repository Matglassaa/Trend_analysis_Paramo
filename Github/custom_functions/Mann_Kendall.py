import numpy as np
import pandas as pd
import scipy.stats as stats

def Mann_Kendall(df, column_name):
    '''
    Function to perform Mann-Kendall trend test on a given time series data.
    '''
    #
    if isinstance(df, pd.Series):
        x = df.to_numpy()
        name = df.name if df.name else "Series"
    else:
        x = df[column_name].to_numpy()
        name = column_name
    

    #create a new array by using a mask
    non_nan_mask = ~np.isnan(x)
    x = x[non_nan_mask]

    n = len(x)
    S = 0

    for k in range(n-1):
        for j in range(k+1, n):
            if x[j] > x[k]:
                S += 1
            elif x[j] < x[k]:
                S -= 1
            # if x[j] == x[k], S is unchanged
    # Uique values and their counts
    unique_x, tp = np.unique(x, return_counts=True)     
    g = len(unique_x)   #number of tied groups

    # Calculate the variance of S
    var_S = (n*(n-1)*(2*n+5) - np.sum(tp*(tp-1)*(2*tp+5))) / 18

    #compute Z
    if S > 0:
        Z = (S-1)/np.sqrt(var_S)
    elif S < 0:
        Z = (S+1)/np.sqrt(var_S)
    else:
        Z = 0
    
    # Calculate the p-value
    p = 2 * (1 - stats.norm.cdf(abs(Z)))  # two-sided p-value

    # Sen's slope estimator
    slopes = []
    for k in range(n-1):
        for j in range(k+1, n):
            slopes.append((x[j] - x[k]) / (j - k))
    sen_slope = np.median(slopes)

    #print results
    print(f'Mann-Kendall Test Results for {column_name}')
    print("="*50)
    print(f'S = {S}')
    print(f'Var(S) = {var_S}')
    print(f'p = {p}')
    if Z > 0:
        print(f'Trend: Increasing')
    elif Z < 0:
        print(f'Trend: Decreasing')
    print("-"*50)
    print(f"Sen's slope: {sen_slope:.2e} K/day, or {sen_slope*365.25:.2e} K/year")
    print("-"*50)
    
    return S, Z, p, sen_slope

import numpy as np
import pandas as pd
from scipy import stats

def Seasonal_Mann_Kendall(data, column_name=None, period=12):
    '''
    Perform the Seasonal Mann-Kendall trend test (Hirsch et al. 1982)
    on a given time series (monthly, quarterly, etc.)

    Parameters
    ----------
    data : pandas DataFrame or pandas Series
        If DataFrame, must have a DateTimeIndex and the column specified.
        If Series, must have a DateTimeIndex.
    column_name : str, optional
        Column to test (required if data is a DataFrame)
    period : int, optional
        Number of seasons (12 for monthly, 4 for quarterly, etc.)

    Returns
    -------
    S_total : float
        Mann-Kendall S statistic
    Z : float
        Normalized test statistic
    p : float
        p-value
    overall_slope : float
        Median Sen's slope
    seasonal_slopes : dict
        Sen's slope for each season
    '''

    # --- handle both Series and DataFrame ---
    if isinstance(data, pd.Series):
        series = data.dropna()
        name = series.name or "Series"
    elif isinstance(data, pd.DataFrame):
        if column_name is None:
            raise ValueError("column_name must be provided when input is a DataFrame.")
        if column_name not in data.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")
        series = data[column_name].dropna()
        name = column_name
    else:
        raise TypeError("Input must be a pandas DataFrame or Series.")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Input must have a DateTimeIndex.")

    S_total, Var_total = 0, 0
    all_slopes = []
    seasonal_slopes = {}

    # --- loop through each season (e.g. months or quarters) ---
    for i in range(period):
        subset = series[series.index.month == i + 1]
        n = len(subset)
        if n < 2:
            continue

        x = subset.to_numpy()
        years = subset.index.year.to_numpy()

        # Mann-Kendall S
        S = 0
        for k in range(n - 1):
            for j in range(k + 1, n):
                if x[j] > x[k]:
                    S += 1
                elif x[j] < x[k]:
                    S -= 1

        # Variance (with ties)
        unique_x, tp = np.unique(x, return_counts=True)
        VarS = (n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18

        S_total += S
        Var_total += VarS

        # Sen's slope for this season
        slopes = []
        for k in range(n-1):
            for j in range(k+1, n):
                # Use actual time difference in years (fractional)
                dt_years = (subset.index[j] - subset.index[k]).days / 365.25
                if dt_years == 0:
                    continue  # skip same-day comparisons
                slope = (x[j] - x[k]) / dt_years
                slopes.append(slope)
        if slopes:
            sen_slope = np.median(slopes)
            seasonal_slopes[i + 1] = sen_slope
            all_slopes.extend(slopes)

    # --- overall Z ---
    if S_total > 0:
        Z = (S_total - 1) / np.sqrt(Var_total)
    elif S_total < 0:
        Z = (S_total + 1) / np.sqrt(Var_total)
    else:
        Z = 0

    # --- p-value ---
    p = 2 * (1 - stats.norm.cdf(abs(Z)))

    # --- overall Sen's slope ---
    overall_slope = np.median(all_slopes) if all_slopes else np.nan

    # --- Print results ---
    print(f"Seasonal Mann-Kendall Test Results for {name}")
    print("=" * 60)
    print(f"S = {S_total}")
    print(f"Var(S) = {Var_total:.2f}")
    print(f"Z = {Z:.3f}")
    print(f"p = {p:.5f}")
    if p < 0.05:
        if Z > 0:
            print("Trend: Increasing (significant)")
        else:
            print("Trend: Decreasing (significant)")
    else:
        print("Trend: No significant trend")
    print("-" * 60)
    print(f"Overall Sen's slope: {overall_slope:.2e} units/year")
    print("Seasonal Sen's slopes (units/year):")
    for season, slope in seasonal_slopes.items():
        print(f"  Season {season}: {slope:.3f}")
    print("-" * 60)

    return S_total, Z, p, overall_slope, seasonal_slopes


# def Seasonal_Mann_Kendall(df, column_name, period=12):
#     '''
#     Perform the Seasonal Mann-Kendall trend test (Hirsch et al. 1982)
#     on a given time series (monthly, quarterly, etc.)

#     Parameters
#     ----------
#     data : pandas DataFrame or pandas Series
#         If DataFrame, must have a DateTimeIndex and the column specified.
#         If Series, must have a DateTimeIndex.
#     column_name : str, optional
#         Column to test (required if data is a DataFrame)
#     period : int, optional
#         Number of seasons (12 for monthly, 4 for quarterly, etc.)

#     Returns
#     -------
#     S_total : float
#         Mann-Kendall S statistic
#     Z : float
#         Normalized test statistic
#     p : float
#         p-value
#     overall_slope : float
#         Median Sen's slope
#     seasonal_slopes : dict
#         Sen's slope for each season
#     '''

#     # --- handle both Series and DataFrame ---
#     if isinstance(df, pd.Series):
#         series = df.dropna()
#         name = series.name or "Series"
#     elif isinstance(df, pd.DataFrame):
#         if column_name is None:
#             raise ValueError("column_name must be provided when input is a DataFrame.")
#         if column_name not in df.columns:
#             raise ValueError(f"Column '{column_name}' not found in DataFrame.")
#         series = df[column_name].dropna()
#         name = column_name
#     else:
#         raise TypeError("Input must be a pandas DataFrame or Series.")

#     if not isinstance(series.index, pd.DatetimeIndex):
#         raise TypeError("Input must have a DateTimeIndex.")
#     S_total, Var_total = 0, 0
#     all_slopes = []
#     seasonal_slopes = {}

#     # loop through each season (e.g. months)
#     for i in range(period):
#         subset = series[series.index.month == i+1]
#         n = len(subset)
#         if n < 2:
#             continue

#         x = subset.to_numpy()
#         years = subset.index.year.to_numpy()

#         # Mann-Kendall S
#         S = 0
#         for k in range(n-1):
#             for j in range(k+1, n):
#                 if x[j] > x[k]:
#                     S += 1
#                 elif x[j] < x[k]:
#                     S -= 1

#         # Variance with ties
#         unique_x, tp = np.unique(x, return_counts=True)
#         VarS = (n*(n-1)*(2*n+5) - np.sum(tp*(tp-1)*(2*tp+5))) / 18

#         S_total += S
#         Var_total += VarS

#         # Sen's slope for this season
#         slopes = []
#         for k in range(n-1):
#             for j in range(k+1, n):
#                 slope = (x[j] - x[k]) / (years[j] - years[k])
#                 slopes.append(slope)
#         if slopes:
#             sen_slope = np.median(slopes)
#             seasonal_slopes[i+1] = sen_slope
#             all_slopes.extend(slopes)

#     # overall Z
#     if S_total > 0:
#         Z = (S_total - 1) / np.sqrt(Var_total)
#     elif S_total < 0:
#         Z = (S_total + 1) / np.sqrt(Var_total)
#     else:
#         Z = 0

#     # p-value
#     p = 2 * (1 - stats.norm.cdf(abs(Z)))

#     # overall Sen's slope
#     overall_slope = np.median(all_slopes) if all_slopes else np.nan

#     # Print results
#     print(f"Seasonal Mann-Kendall Test Results for {column_name}")
#     print("="*60)
#     print(f"S = {S_total}")
#     print(f"Var(S) = {Var_total:.2f}")
#     print(f"Z = {Z:.3f}")
#     print(f"p = {p:.5f}")
#     if p < 0.05:
#         if Z > 0:
#             print("Trend: Increasing (significant)")
#         else:
#             print("Trend: Decreasing (significant)")
#     else:
#         print("Trend: No significant trend")
#     print("-"*60)
#     print(f"Overall Sen's slope: {overall_slope:.2e} K/year")
#     print("Seasonal Sen's slopes (units/year):")
#     for season, slope in seasonal_slopes.items():
#         print(f"  Season {season}: {slope:.3f}")
#     print("-"*60)

#     return S_total, Z, p, overall_slope, seasonal_slopes