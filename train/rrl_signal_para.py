# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from signals.sharpe_ratio import sharpe_ratio
from signals.MA import calc_ma
from signals.MTM import calc_mtm
from signals.OBV import calc_obv
import random


# turn [2,3,4] to [3,4,0]
def move_front(X):
    new = list(X)[1:]
    new.append(0)
    return new


def move_behind(X):
    new = [0]
    extra = X[:-1]
    new.extend(extra)
    return new


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
    ft = diff_price_behind(Ft)
    new_x = list(X)[1:]
    new_x.append(0)
    # price_return = miu * (np.array(Ft) * np.array(rt) - np.abs(ft) * np.array(new_x) * delta)
    price_return = miu * (np.array(Ft) * np.array(rt) - np.abs(ft) * delta)
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


def reward_function(X, miu, delta, M, para, threshold, time_index, origin_asset, close, vol):
    Ft = update_position(X, para, threshold, close, vol)
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


def calc_dSdw(X, miu, delta, M, theta, threshold, time_index, origin_asset, close, vol):
    Ret, sharp, asset_list, Ft, pnl, day_index = reward_function(X, miu, delta, M, theta, threshold, time_index, origin_asset, close, vol)
    print("asset")
    print(asset_list[0], asset_list[-1])

    diff_position = list()
    for temp in range(len(Ft) - 1):
        if Ft[temp + 1] != Ft[temp]:
            diff_position.append(1)
        else:
            diff_position.append(0)
    diff_position.append(0)
    print("position", sum(diff_position))

    ret = Ret[M:]
    # print("ret")
    # print(ret)
    ret = np.array(ret)
    ret_squre = [temp * temp for temp in ret]
    A = np.mean(ret)
    B = np.mean(ret_squre)
    S = A / np.sqrt(B - A ** 2)
    dSdA = S * (1 + S ** 2) / A
    dSdB = -0.5 * (S ** 3) / (A ** 2)
    dAdR = 1.0 / len(ret)
    dBdR = 2.0 * ret / len(ret)
    dRdF = miu * diff_price_behind(X)[M:] + list(miu * delta * np.array(move_front(X)[M:]) * np.sign(diff_price_behind(Ft)[M:]))
    dRdFp = -miu * delta * np.array(move_front(X)[M:]) * np.sign(diff_price_behind(Ft)[M:])
    dSdw = np.zeros(M+2)

    rt = diff_price_front(X)
    dFdw = [0] * (M + 2)
    xt = [1]
    xt.extend(rt[0:M])
    xt.append(Ft[M])
    xt = np.array(xt)
    dFpdw = (1 - np.tanh(sum(theta * xt)) ** 2) * (xt + [ele * theta[-1] for ele in dFdw])
    for temp_i in range(len(ret) - 2):
        dFdw = dFpdw
        xt = [1]
        xt.extend(rt[temp_i + 1: temp_i + 1 + M])
        xt.append(Ft[temp_i + 1 + M])
        xt = np.array(xt)
        dFpdw = (1 - np.tanh(sum(theta * xt)) ** 2) * (xt + [ele * theta[-1] for ele in dFdw])

        dSdw += (dSdA * dAdR + dSdB * dBdR[temp_i + 2]) * (dRdF[temp_i + 2] * dFdw + dRdFp[temp_i + 2] * dFpdw)
        # print("here", dSdw)

    # add rand
    # temp_list = list()
    # for item in range(M + 2):
    #     temp_list.append((random.random() - 0.5))
    # theta = np.array(temp_list)

    # print("gradient", dSdw + theta * 0.1)


    return dSdw, sharp


def update_para(para, rho, dSdw):
    print('dSdw', dSdw)
    return para + rho * dSdw


def update_position(origin_data, para, threshold, close, vol, window=288, ma1=72, ma2=288, ma3=288*3):
    m = 5
    mtm_list = calc_mtm(close, window)
    mtm_list = (mtm_list - np.mean(mtm_list)) / np.std(mtm_list)
    ma1_list = calc_ma(origin_data, ma1)
    ma1_list = (ma1_list - np.mean(ma1_list)) / np.std(ma1_list)
    ma2_list = calc_ma(origin_data, ma2)
    ma2_list = (ma2_list - np.mean(ma2_list)) / np.std(ma2_list)
    ma3_list = calc_ma(origin_data, ma3)
    ma3_list = (ma3_list - np.mean(ma3_list)) / np.std(ma3_list)
    obv_list = calc_obv(close, vol)
    obv_list = (obv_list - np.mean(obv_list)) / np.std(obv_list)
    Ft = [0]

    mid_variables = np.transpose(np.vstack([mtm_list, ma1_list, ma2_list, ma3_list, obv_list]))
    for temp_i in range(1, len(origin_data)):
        Xt = [1]
        Xt.extend(mid_variables[temp_i])
        Xt.append(Ft[temp_i - 1])
        new_value = np.tanh(sum(Xt * para))
        if new_value > threshold:
            Ft.append(1)
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
    p_close = gdax['close']
    p_open = gdax['open']
    p_volumn = gdax['vol']
    time_str = gdax['time']
    time_list = [pd.to_datetime(temp) for temp in time_str.values]
    # using mid price
    price_mid = np.array((p_close + p_open) / 2.0)

    price_mid = (price_mid - np.mean(price_mid)) / np.std(price_mid)

    new_df_values = np.array([price_mid, time_list])
    new_df_values = new_df_values.transpose()
    price_df = pd.DataFrame(new_df_values, columns=['price', 'time'], index=p_close.index)
    train_length = int(len(price_df) * 2 / 3)
    price_train_all = price_df[:train_length]
    price_train = price_train_all['price']
    time_train = price_train_all['time'].values
    price_test_all = price_df[train_length:]
    price_test = price_test_all['price']
    time_test = price_test_all['time'].values

    price_close = p_close[:train_length].values
    volume = p_volumn[:train_length].values


    # Initializing parameters
    m = 5  # The number of past r_t that would used to represent x_t
    data_train = np.array(price_train.values)
    temp_list = list()
    for item in range(m + 2):
        temp_list.append((random.random() - 0.5))
    theta = np.array(temp_list) * 10
    theta = np.array([ 3.27987256,  4.95639962, -3.31348798, -1.78083594, -3.53054956, -1.13249043,-0.13245841])


    print(theta)
    delta = 0.001  # transaction fee
    miu = 1  # how much to buy
    threshold = 0.4

    origin_capital = 500

    # First iterate
    price_return, sharpe, asset, position, pnl, day_index = \
        reward_function(data_train, miu, delta, m, theta, threshold, time_train, origin_capital, price_close, volume)
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
    df.to_csv("../signal_results/real_test.csv")



    # Solve params
    max_loop = 500
    converged = False
    epsilon = 1e-3
    rate = 0.01

    print("begin loop!")

    for loop_i in range(max_loop):
        print(loop_i)
        if loop_i == max_loop - 1:
            print("timeout")
        if converged:
            break
        partial_d, sharpe = \
            calc_dSdw(data_train, miu, delta, m, theta, threshold, time_train, origin_capital, price_close, volume)
        print(partial_d)
        if cal_mod(partial_d) < epsilon:
            converged = True
            good_para = theta
        else:
            theta = update_para(theta, rate, partial_d)
            print('theta', theta)
            print('sharpe', sharpe)

        # write sharpe
        f_sharpe = open("../signal_results/sharpe_log_para.txt", 'a')
        f_sharpe.write(str(sharpe) + '\n')
        f_sharpe.close()

    print(good_para)




if __name__ == "__main__":
    main()
