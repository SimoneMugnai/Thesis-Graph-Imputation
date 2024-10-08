import numpy as np
import pandas as pd
from scipy.signal import gaussian


def ensure_list(obj):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    else:
        return [obj]


def prediction_dataframe(y, index, columns=None, aggregate_by='mean'):
    """Aggregate batched predictions in a single DataFrame."""
    df = pd.DataFrame(data=y, index=index, columns=columns)
    preds_by_step = df.groupby(df.index)

    if aggregate_by == 'mean':
        df_agg = preds_by_step.mean()
    elif aggregate_by == 'sd':
        df_agg = preds_by_step.std()
    elif aggregate_by == 'central':
        df_agg = preds_by_step.aggregate(lambda x: x.iloc[len(x) // 2])
    elif aggregate_by == 'smooth_central':
        df_agg = preds_by_step.aggregate(
            lambda x: np.average(x, weights=gaussian(len(x), 1)))
    elif aggregate_by == 'last':
        df_agg = preds_by_step.aggregate(lambda x: x.iloc[-1])
    else:
        raise ValueError(
            f'Invalid aggregation method. Choose from mean, central, smooth_central, last.')

    return df_agg


def prediction_dataframe_v2(df, aggregate_by='mean'):
    """Aggregate batched predictions in a single DataFrame."""
    preds_by_step = df.groupby(df.index)

    aggregation_methods = {
        'mean': lambda x: x.mean(),
        'sd': lambda x: x.std(),
        'central': lambda x: x.apply(lambda y: y.iloc[len(y) // 2]),
        'smooth_central': lambda x: x.apply(
            lambda y: np.average(y, weights=gaussian(len(y), 1))),
        'last': lambda x: x.apply(
            lambda y: y.iloc[-1] if len(y) > 0 else np.nan),
        'trimmed_mean': lambda x: x.apply(
            lambda y: y.iloc[1:-1].mean() if len(y) > 4 else y.mean())
    }

    aggregate_by = ensure_list(aggregate_by)  # Ensure aggregate_by is a list
    results = {}

    for method in aggregate_by:
        if method in aggregation_methods:
            results[method] = aggregation_methods[method](preds_by_step)
        else:
            raise ValueError(f'Invalid aggregation method "{method}". '
                             f'Choose from {list(aggregation_methods.keys())}.')

    return results


def prediction_dataframe_v3(df, imputation_window, lags,
                            aggregate_by='mean'):
    assert len(df) % imputation_window == 0, \
        ('Improper imputation window. Number of samples must be '
         'divisible by the imputation window.')
    n_samples = len(df) // imputation_window
    ll = np.tile(np.arange(imputation_window, 0, -1), n_samples)

    aggregation_methods = {
        'mean': lambda x: x.mean(),
        'sd': lambda x: x.std(),
        'central': lambda x: x.apply(lambda y: y.iloc[len(y) // 2]),
        'smooth_central': lambda x: x.apply(
            lambda y: np.average(y, weights=gaussian(len(y), 1))),
        'last': lambda x: x.apply(
            lambda y: y.iloc[-1] if len(y) > 0 else np.nan),
        'trimmed_mean': lambda x: x.apply(
            lambda y: y.iloc[1:-1].mean() if len(y) > 4 else y.mean())
    }

    aggregate_by = ensure_list(aggregate_by)  # Ensure aggregate_by is a list
    results = {aggr_by: dict() for aggr_by in aggregate_by}

    index = df.index.unique().sort_values()

    for lag in lags:
        assert lag > 0, f'Invalid lag value {lag}.'
        # Compute the mean of the last `lag` predictions
        # Time step of interest will be lag steps behind the last time step
        df_lag = df[ll <= lag]
        groups = df_lag.groupby(df_lag.index)
        for method in aggregate_by:
            df_aggr = aggregation_methods[method](groups)
            results[method][lag] = df_aggr.reindex(index)
    return results


def calculate_residuals(actual, imputed):
    # Compute residuals (actual - imputed)
    residuals = actual - imputed

    # Ensure residuals are positive by taking the absolute value
    residuals = residuals.abs()

    # Replace NaN values with 0
    residuals = residuals.fillna(0)

    return residuals
