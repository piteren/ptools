"""

 2019 (c) piteren

    stats methods

"""

import pandas as pd
import numpy as np

# mean, std, min, max (from given list of values or np.arr)
def msmx(val_list :list or np.array) -> dict:

    arr = val_list
    if type(val_list) is list: arr = np.asarray(val_list)

    ret_dict = {
        'mean': float(np.mean(arr)),
        'std':  float(np.std(arr)),
        'min':  float(np.min(arr)),
        'max':  float(np.max(arr))}
    ret_dict['string'] = 'mean %.5f, std %.5f, min %.5f, max %.5f'%(ret_dict['mean'],ret_dict['std'],ret_dict['min'],ret_dict['max'])
    return ret_dict

# deep stats (with pandas)
def stats_pd(
        val_list :list,
        n_percentiles=  10) -> str:
    return f'{pd.Series(val_list).describe(percentiles=[0.1*n for n in range(1,n_percentiles)])}'