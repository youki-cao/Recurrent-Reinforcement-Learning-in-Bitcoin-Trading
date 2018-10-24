# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from signals.sharpe_ratio import sharpe_ratio
import random


def diff_price_behind(X):
    diff = list(np.array(X[1:]) - np.array(X[:-1]))
    diff.append(0)
    return diff


def diff_price_front(X):
    diff = [0]
    diff.extend(list(np.array(X[1:]) - np.array(X[:-1])))
    return diff


def calc_return(X, Ft, miu, delta):
    rt = diff_price_behind(X)
    ft = diff_price_front(Ft)
    price_return = miu * (np.array(Ft) * np.array(rt) - delta * np.abs(ft))
    return price_return


def get_date(temp_time):
    year = str(temp_time.year)
    month = str(temp_time.month)
    day = str(temp_time.day)
    return pd.to_datetime(year + '-' + month + '-' + day)


def get_time(temp_time):
    return temp_time.time()


def cal_pnl(asset_list, time_index):
    day_index = list()
    pnl = list()
    for each_time_num in range(len(time_index)):
        if each_time_num == 0:
            day_index.append(get_date(time_index[each_time_num]))
            temp = asset_list[each_time_num]

        elif get_date(time_index[each_time_num]) != get_date(time_index[each_time_num - 1]):
            day_index.append(get_date(time_index[each_time_num]))

        elif each_time_num == len(time_index) - 1:
            pnl.append((asset_list[each_time_num] - temp) / temp)
            continue

        elif get_date(time_index[each_time_num]) != get_date(time_index[each_time_num + 1]):
            pnl.append((asset_list[each_time_num] - temp) / temp)
            temp = asset_list[each_time_num]
    return day_index, pnl


def reward_function(X, miu, delta, M, para, threshold, time_index, origin_asset):
    Ft = update_position(X, para, threshold)
    price_return = calc_return(X, Ft, miu, delta)
    asset_list = list()
    for temp_item in range(len(price_return)):
        if temp_item == 0:
            asset_list.append(origin_asset)
        else:
            asset_list.append(asset_list[temp_item - 1] + price_return[temp_item])
    # Calculate the return of every timestamp
    Ret = list()
    for temp_item in range(len(asset_list)):
        if temp_item == 0:
            Ret.append(0)
        elif asset_list[temp_item - 1] == 0:
            Ret.append(0)
        else:
            Ret.append((asset_list[temp_item] - asset_list[temp_item - 1]) / asset_list[temp_item - 1])
    day_index, pnl = cal_pnl(asset_list, time_index)
    sharpe = sharpe_ratio(Ret[M:])
    return Ret, sharpe, asset_list, Ft, pnl, day_index


def calc_dSdw(X, miu, delta, M, theta, threshold, time_index, origin_asset):
    Ret, sharp, asset_list, Ft, pnl, day_index = reward_function(X, miu, delta, M, theta, threshold, time_index, origin_asset)

    diff_position = list()
    for temp in range(len(Ft) - 1):
        if Ft[temp + 1] != Ft[temp]:
            diff_position.append(1)
        else:
            diff_position.append(0)
    diff_position.append(0)
    print("position", sum(diff_position))

    ret = Ret[M:]
    print("ret", ret)
    ret = np.array(ret)
    ret_squre = [temp * temp for temp in ret]
    A = np.mean(ret)
    B = np.mean(ret_squre)
    S = A / np.sqrt(B - A ** 2)
    dSdA = S * (1 + S ** 2) / A
    dSdB = -0.5 * (S ** 3) / (A ** 2)
    dAdR = 1.0 / len(ret)
    dBdR = 2.0 * ret / len(ret)
    dRdF = -miu * delta * np.sign(diff_price_behind(Ft)[M:])
    dRdFp = miu * diff_price_behind(X)[M:] + list(miu * delta * np.sign(-1 * diff_price_behind(Ft)[M:]))
    dFdw = np.zeros(M+2)
    dFpdw = np.zeros(M+2)
    dSdw = np.zeros(M+2)
    for i in range(len(ret) - 1, -1 ,-1):
        if i != len(ret) - 1:
            dFpdw = dFdw.copy()
        dFdw = (1 - Ft[i]**2) * (X[i] + theta[M+2-1] * dFpdw)
        dSdw += (dSdA * dAdR + dSdB * dBdR[i]) * (dRdF[i] * dFdw + dRdFp[i] * dFpdw)
    print("gradient", dSdw)
    return dSdw, sharp


def update_para(para, rho, dSdw):
    print('dSdw', dSdw)
    return para + rho * dSdw


def update_position(origin_data, para, threshold=0.6):
    m = len(para) - 2
    Ft = [0] * (m - 1)
    Xt = [0] * (m + 2)
    for temp_i in range(1, len(origin_data) + 2 - m):
        Xt[0] = 1
        Xt[1:m + 1] = origin_data[temp_i - 1:temp_i + m - 1]
        Xt[m + 1] = Ft[temp_i - 1]
        # print("sum", sum(Xt * para))
        new_value = np.tanh(sum(Xt * para))
        # s = sum(Xt * para)
        # new_value = max(0, s) + 0.3 * min(0, s)
        # print("new valeeue", new_value)
        if new_value > threshold:
            Ft.append(1)
        # elif new_value < -1 * threshold:
        #     Ft.append(-1)
        else:
            Ft.append(0)
    position_Ft = [0]
    position_Ft.extend(Ft)
    position_Ft.pop()
    return position_Ft


def cal_mod(vector):
    return np.sqrt(np.sum(np.array(vector) ** 2))


def main():
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
    price_train = price_train_all['price']
    time_train = price_train_all['time'].values
    price_test_all = price_df[train_length:]
    price_test = price_test_all['price']
    time_test = price_test_all['time'].values

    # Initializing parameters
    m = 5  # The number of past r_t that would used to represent x_t
    data_train_raw = np.array(price_train.values)
    data_train = (data_train_raw - np.mean(data_train_raw)) / np.std(data_train_raw)
    temp_list = list()
    for item in range(m + 2):
        temp_list.append((random.random() - 0.5))
    theta = np.array(temp_list) * 10

    print(theta)
    delta = 0.001  # transaction fee
    miu = 1  # how much to buy
    threshold = 0.4
    origin_capital = 10

    # First iterate
    price_return, sharpe, asset, position, pnl, day_index = reward_function(data_train, miu, delta, m, theta, threshold, time_train, origin_capital)
    """
    price_return += 1
    total_return = 1
    for temp_i in range(1, len(price_return)):
        total_return *= price_return[temp_i]
    """
    print("asset:", asset[-1])
    print("sharp:", sharpe)
    diff_position = list()
    for temp in range(len(position) - 1):
        if position[temp + 1] != position[temp]:
            diff_position.append(1)
        else:
            diff_position.append(0)
    diff_position.append(0)
    print(sum(diff_position))
    value = np.transpose([price_return, asset, position, data_train, diff_position])
    df = pd.DataFrame(value, columns=['ret', 'asset', 'position', 'price', 'diff'])
    df.to_csv("../tmp/real_test.csv")



    # Solve params
    max_loop = 500
    converged = False
    epsilon = 1e-3
    rate = 0.5

    print("begin loop!")

    for loop_i in range(max_loop):
        if loop_i == max_loop - 1:
            print("timeout")
        if converged:
            break
        partial_d, sharpe = calc_dSdw(data_train, miu, delta, m, theta, threshold, time_train, origin_capital)
        print(partial_d)
        if cal_mod(partial_d) < epsilon:
            converged = True
            good_para = theta
        else:
            theta = update_para(theta, rate, partial_d)
            print('theta', theta)
            print('sharpe', sharpe)

    print(good_para)




if __name__ == "__main__":
    main()

