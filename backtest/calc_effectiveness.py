import pandas as pd
import numpy as np
from signals.annual_return import annual_return
from signals.hit_ratio import hit_ratio
from signals.max_drawdown import max_drawdown


def calc_effectiveness(file_path):
    df = pd.read_csv(file_path)
    pnl = df['0'].values
    # calculate asset
    origin = 20000
    asset = list()
    for temp_i in range(len(pnl)):
        if temp_i == 0:
            asset.append(origin * (1 + pnl[temp_i]))
        else:
            asset.append(asset[-1] * (1 + pnl[temp_i]))
    ann_return = annual_return(asset)
    hit = hit_ratio(pnl)
    mdd = max_drawdown(asset)
    print("annual return", ann_return)
    print("hit ratio", hit)
    print("maximum drawdown", mdd)
    pass

