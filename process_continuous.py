import pickle
import os
import torch
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from scipy import interpolate as inter

from Eyedata import EyeData

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

csv_path = "./data/continuous/csv"
pkl_path = "./data/continuous/pkl"

sample_rate = 250

def standardize(data):
    return (data - np.mean(data))/np.std(data)

def minmax_normalize(data):
    return (data - np.min(data))/(np.max(data) - np.min(data))

def lpf(data, lowcut, order=2):
    b, a = butter(order, 2*lowcut/sample_rate, 'lowpass')
    return filtfilt(b, a, data)

def process_csv(csv_file):
    data = pd.read_csv(csv_file)
    vl = minmax_normalize(data['CH1'].values)
    vr = minmax_normalize(data['CH3'].values)
    hl = minmax_normalize(data['CH2'].values)
    hr = minmax_normalize(data['CH4'].values)
    
    delta_v = vr - vl
    delta_h = hr - hl
    value = {'vr':vr, 'vl':vl, 'hr':hr, 'hl':hl, 'dv':delta_v, 'dh':delta_h}
    offset = {'vr':vr, 'vl':vl, 'hr':hr, 'hl':hl, 'dv':delta_v, 'dh':delta_h}
    std = {'vr':vr, 'vl':vl, 'hr':hr, 'hl':hl, 'dv':delta_v, 'dh':delta_h}
    
    for k, m in value.items(): 
        value[k] = lpf(m, 5, 3)
        offset[k] = lpf(m, 0.08, 3)
        std[k] = value[k] - offset[k]

    return value, offset, std

def process_data(csv_file, pkl_file, blink_period, press_delay):
    # print(csv_file, pkl_file)
    _ , _ , std = process_csv(csv_file)
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    temp = EyeData()
    result = []
    for i in range(len(data)-1):
        temp = data[i]
        start, end = temp.start_cnt, temp.end_cnt
        
        pos_x = np.linspace(temp.pos_x[0], temp.pos_x[-1], end-start)
        g = inter.interp1d(temp.pos_x, temp.pos_y, kind ="linear")
        pos_y = g(pos_x)
        pos_x = (pos_x - 1280) / 2560 
        pos_y = (pos_y - 720) / 1440
        
        blink_label = np.zeros_like(pos_x)
        
        for label in temp.blink_time:
            start_blink = int(label-press_delay*250) - start
            end_blink = int(label-(press_delay-blink_period)*250) - start
            # print(f"{i}: {temp.blink_time}")
            # print(start_blink, end_blink)
            blink_label[start_blink:end_blink] = 1
            
        vl, vr, hl, hr = std['vl'][start:end], std['vr'][start:end], std['hl'][start:end], std['hr'][start:end]

        frame = np.stack([vl, vr, hl, hr, pos_x, pos_y, blink_label], axis=-1)
        # print(frame.shape)
        df = pd.DataFrame(frame)
        result.append(df)
    result = pd.concat(result)
    return result

if __name__ == "__main__":
    processed = []
    # for j in range(15):
    #     processed.append(process_data(f"{csv_path}/Device_{j}_Volts.csv", f"{pkl_path}/Eyedata_{j}.pkl", 0.3, 0.1))
        
    # df = pd.concat(processed)
    # df.rename(columns={0:'vl', 1:'vr', 2:'hl', 3:'hr', 4:'pos_x', 5:'pos_y', 6:'blink_label'}, inplace=True)
    # print(df)
    # df.to_csv("./data/continuous/continuous_train.csv", index=False)
    
    for j in range(15, 17):
        processed.append(process_data(f"{csv_path}/Device_{j}_Volts.csv", f"{pkl_path}/Eyedata_{j}.pkl", 0.3, 0.1))
        
    df = pd.concat(processed)
    df.rename(columns={0:'vl', 1:'vr', 2:'hl', 3:'hr', 4:'pos_x', 5:'pos_y', 6:'blink_label'}, inplace=True)
    print(df)
    df.to_csv("./data/continuous/continuous_test.csv", index=False)