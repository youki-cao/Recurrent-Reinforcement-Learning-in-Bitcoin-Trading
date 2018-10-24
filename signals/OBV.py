import numpy as np
import pandas as pd


# input: list, list
# output: list
# default: OBV[0] = 1
# Calculate the
def calc_obv(close_price, volume):
    obv = [0]
    for temp_i in range(1, len(close_price)):
        if close_price[temp_i] > close_price[temp_i - 1]:
            obv.append(obv[temp_i - 1] + volume[temp_i])
        elif close_price[temp_i] == close_price[temp_i - 1]:
            obv.append(obv[temp_i - 1])
        else:
            obv.append(obv[temp_i - 1] - volume[temp_i])
    return obv
