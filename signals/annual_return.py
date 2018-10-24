import numpy as np


# input : list
# output: annual_return, list
def annual_return(asset):
    all_return = (asset[-1] - asset[0]) / asset[0]
    return all_return / len(asset) * 365
