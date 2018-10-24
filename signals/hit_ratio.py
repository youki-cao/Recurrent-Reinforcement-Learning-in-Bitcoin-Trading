import numpy as np


# input : pnl list
# output: hit_ratio, float
def hit_ratio(pnl):
    hit_num = sum([1 if temp > 0 else 0 for temp in pnl])
    all_num = sum([1 if temp > 0 or temp < 0 else 0 for temp in pnl])
    return hit_num / all_num
