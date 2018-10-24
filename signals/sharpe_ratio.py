import numpy as np
import pandas as pd
from math import sqrt


# input : list
# output: sharpe ratio, float
def sharpe_ratio(ret):
    mean_ret = np.mean(ret)
    std_ret = np.std(ret)
    if std_ret == 0:
        return 0
    sharpe = sqrt(365) * mean_ret / std_ret
    return sharpe


