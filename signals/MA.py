import pandas as pd
import numpy as np


# input: list
# output: list
def calc_ma(price, day):
    ma_list = list()
    for temp_i in range(len(price)):
        ma_list.append(np.mean(price[max(0, temp_i - day + 1): temp_i + 1]))
    return ma_list
