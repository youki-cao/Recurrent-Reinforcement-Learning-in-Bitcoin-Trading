# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from signals.sharpe_ratio import sharpe_ratio
import matplotlib.pyplot as plt
from train.rrl_origin import reward_function, calc_return, cal_pnl


theta = np.array([16, -7, -3, -1, 0.5, 2, 1])
m = 5
delta = 0.001  # transaction fee
miu = 1  # how much to buy
threshold = 0.4
origin_capital_normal = 10
origin_capital = 20000


# input: df, columns=['price', 'time']
def get_position(price):
    data = np.array(price['price'].values)
    time = price['time'].values
    data_normal = (data - np.mean(data)) / np.std(data)
    price_return_, sharpe_, asset_, position, pnl_, day_index = reward_function(data_normal, miu, delta, m, theta, threshold, time, origin_capital_normal)

    price_return = calc_return(data, position, miu, delta)
    asset_list = list()
    for temp_item in range(len(price_return)):
        if temp_item == 0:
            asset_list.append(origin_capital)
        else:
            asset_list.append(asset_list[temp_item - 1] + price_return[temp_item])

    Ret = list()
    for temp_item in range(len(asset_list)):
        if temp_item == 0:
            Ret.append(0)
        elif asset_list[temp_item - 1] == 0:
            Ret.append(0)
        else:
            Ret.append((asset_list[temp_item] - asset_list[temp_item - 1]) / asset_list[temp_item - 1])
    print(type(day_index[0]))
    day_index, pnl = cal_pnl(asset_list, time)

    sharpe = sharpe_ratio(pnl)
    print("sharpe", sharpe)

    #output to file
    values = np.transpose(np.array([data, position, asset_list, Ret]))
    output_df = pd.DataFrame(values, index=time)
    output_df.to_csv("../results/result_origin/result_test.csv")

    output2 = pd.DataFrame(pnl, index=day_index)
    output2.to_csv("../results/result_origin/results_daily.csv")

    return time, data, position, asset_list, Ret, pnl, day_index


def get_benchmark(price):
    data = np.array(price['price'].values)
    time = price['time'].values
    data_normal = (data - np.mean(data)) / np.std(data)
    price_return_, sharpe_, asset_, position, pnl_, day_index = reward_function(data_normal, miu, delta, m, theta, threshold, time, origin_capital_normal)

    # calculate in the origin price series
    position = [1] * len(position)

    price_return = calc_return(data, position, miu, delta)
    asset_list = list()
    for temp_item in range(len(price_return)):
        if temp_item == 0:
            asset_list.append(origin_capital)
        else:
            asset_list.append(asset_list[temp_item - 1] + price_return[temp_item])

    Ret = list()
    for temp_item in range(len(asset_list)):
        if temp_item == 0:
            Ret.append(0)
        elif asset_list[temp_item - 1] == 0:
            Ret.append(0)
        else:
            Ret.append((asset_list[temp_item] - asset_list[temp_item - 1]) / asset_list[temp_item - 1])
    print(type(day_index[0]))
    day_index, pnl = cal_pnl(asset_list, time)

    sharpe = sharpe_ratio(pnl)
    print("sharpe", sharpe)

    #output to file
    values = np.transpose(np.array([data, position, asset_list, Ret]))
    output_df = pd.DataFrame(values, index=time)
    output_df.to_csv("../results/result_origin/result_test_bmm.csv")

    output2 = pd.DataFrame(pnl, index=day_index)
    output2.to_csv("../results/result_origin/results_daily_bmm.csv")

    return time, data, position, asset_list, Ret, pnl, day_index

gdax = pd.read_csv('../market_data/gdax.csv', index_col=0)
close = gdax['close']
open = gdax['open']
time_str = gdax['time']
time_list = [pd.to_datetime(temp) for temp in time_str.values]
# using mid price
price_mid = (close + open) / 2.0
new_df_values = np.array([price_mid.values, time_list])
new_df_values = new_df_values.transpose()
price_df = pd.DataFrame(new_df_values, columns=['price', 'time'], index=price_mid.index)
train_length = int(len(price_df) * 2 / 3)
price_train_all = price_df[:train_length]
price_test_all = price_df[train_length:]
# price_test_all = price_test_all[:35713]
time, data, position, asset_list, Ret, pnl, day_index  = get_position(price_test_all)
time_b, data_b, position_b, asset_list_b, Ret_b, pnl_b, day_index_b = get_benchmark(price_test_all)

values_df = np.transpose(np.array([asset_list, asset_list_b]))
plt_df = pd.DataFrame(values_df, index=time)
plt.figure()
plt_df[0].plot(label='RRL')
plt_df[1].plot(label='Buy&Hold')
plt.ylabel('asset')
plt.xlabel('time')
plt.legend()
plt.show()
# plt.savefig('../results/result_origin/asset_test_roll.png')

daily_df = np.transpose(np.array([pnl, pnl_b]))
daily_df = pd.DataFrame(daily_df, index=day_index)
plt.figure()
daily_df[0].plot(label='RRL')
daily_df[1].plot(label='Buy&Hold')
plt.ylabel('p&l')
plt.xlabel('time')
plt.legend()
plt.show()
