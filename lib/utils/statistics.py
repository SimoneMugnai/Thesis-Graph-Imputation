import pandas as pd
import numpy as np

def ensure_list(obj):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    else:
        return [obj]






def prediction_dataframe(y, index, columns=None, aggregate_by='mean'):
    """Aggregate batched predictions in a single DataFrame.

    @param (list or np.ndarray) y: the list of predictions.
    @param (list or np.ndarray) index: the list of time indexes coupled with the predictions.
    @param (list or pd.Index) columns: the columns of the returned DataFrame.
    @param (str or list) aggregate_by: how to aggregate the predictions in case there are more than one for a step.
    - `mean`: take the mean of the predictions
    - `central`: take the prediction at the central position, assuming that the predictions are ordered chronologically
    - `smooth_central`: average the predictions weighted by a gaussian signal with std=1
    - `last`: take the last prediction
    @return: pd.DataFrame df: the evaluation mask for the DataFrame
    """
    index = [np.full((y[i].shape[0],), index[i], dtype=index.dtype).tolist() for i in range(len(index))]

    #dfs = [pd.DataFrame(data=data.reshape(data.shape[:2]), index=idx, columns=columns) for data, idx in zip(y, index)]
    dfs = [pd.DataFrame(data=y[i].reshape(-1, y[i].shape[-1]), index=idx, columns=columns) 
           for i, idx in enumerate(index)]

    df = pd.concat(dfs)
    preds_by_step = df.groupby(df.index)
    # aggregate according passed methods
    aggr_methods = ensure_list(aggregate_by)
    dfs = []
    for aggr_by in aggr_methods:
        if aggr_by == 'mean':
            dfs.append(preds_by_step.mean())
        if aggr_by == 'sd':
            dfs.append(preds_by_step.std())
    return dfs