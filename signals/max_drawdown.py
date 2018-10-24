import numpy as np


# input : asset list
# output: maximum drawdown, float
def max_drawdown(asset):
    max_price = -9999
    drawdown_list = list()
    for temp_i in range(len(asset)):
        if asset[temp_i] > max_price:
            max_price = asset[temp_i]
        drawdown_list.append(1 - asset[temp_i] / max_price)
    return max(drawdown_list)

