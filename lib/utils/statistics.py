import pandas as pd
import numpy as np

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
            df_agg= preds_by_step.std()
    elif aggregate_by == 'central':
        df_agg = preds_by_step.aggregate(lambda x: x.iloc[len(x) // 2])
    elif aggregate_by == 'smooth_central':
        from scipy.signal import gaussian
        df_agg = preds_by_step.aggregate(lambda x: np.average(x, weights=gaussian(len(x), 1)))
    elif aggregate_by == 'last':
        df_agg = preds_by_step.aggregate(lambda x: x.iloc[-1])
    else:
        raise ValueError(f'Invalid aggregation method. Choose from mean, central, smooth_central, last.')

    return df_agg