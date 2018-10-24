import numpy as np
import pandas as pd


# input: list, int
# output: list

def calc_mtm(price, window):
    mtm_list = list()
    for temp_i in range(len(price)):
        if temp_i < window - 1:
            mtm_list.append(0)
        else:
            mtm_list.append((price[temp_i] - price[temp_i - window + 1]) / price[temp_i])
    return mtm_list
