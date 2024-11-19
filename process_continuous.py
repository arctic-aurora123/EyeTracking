import pickle
import os
import torch
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

from Eyedata import EyeData

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

csv_path = "./data/latest data/csv"
pkl_path = "./data/latest data/pkl"

sample_rate = 250
min_blink_period = 0.1
blink_period, press_delay = 0.4, 0.1

def standardize(data):
    return (data - np.mean(data))/np.std(data)

def minmax_normalize(data):
    return (data - np.min(data))/(np.max(data) - np.min(data))

def lpf(data, lowcut, order=2):
    b, a = butter(order, 2*lowcut/sample_rate, 'lowpass')
    return filtfilt(b, a, data)

def plot_signal(start, end, signal):
    end *= 250
    time = np.arange(start, end+1)
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.plot(time, signal['vl'], label='左眼水平')
    plt.plot(time, signal['vr'], label='右眼水平')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(time, signal['hl'], label='左眼垂直')

    plt.plot(time, signal['hl'], label='右眼垂直')
    plt.legend()
    plt.show()

def process_csv(csv_file):  
    data = pd.read_csv(csv_file)
    
    vl = minmax_normalize(data['CH1'].values)
    vr = minmax_normalize(data['CH2'].values)
    hl = minmax_normalize(data['CH3'].values)
    hr = minmax_normalize(-data['CH4'].values)
    
    value = {'vr':vr, 'vl':vl, 'hr':hr, 'hl':hl}
    offset = {'vr':vr, 'vl':vl, 'hr':hr, 'hl':hl}
    std = {'vr':vr, 'vl':vl, 'hr':hr, 'hl':hl}
    
    for k, m in value.items(): 
        value[k] = lpf(m, 5, 3)
        offset[k] = lpf(m, 0.08, 3)
        std[k] = value[k] - offset[k]

    return value, offset, std
    
def process_data(csv_file, pkl_file, blink_period, press_delay):
    print(csv_file, pkl_file)
    value, _ , std = process_csv(csv_file)
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    temp = EyeData()
    result_std = []
    result_value = []
    
    for i in range(len(data)-1):
        temp = data[i]
        start, end = temp.start_cnt, temp.end_cnt
        print(start, end)
        if pkl_file == './data/latest data/pkl/Eyedata_0.pkl' and i == 1:
            start = 1094
        if pkl_file == './data/latest data/pkl/Eyedata_26.pkl' and i == 1:
            start = 1498
        pos_x = np.linspace(temp.pos_x[0], temp.pos_x[-1], end-start)
        g = interp1d(temp.pos_x, temp.pos_y, kind ="linear")
        pos_y = g(pos_x)
        pos_x = (pos_x - 1280) / 2560 
        pos_y = (pos_y - 720) / 1440
        
        blink_label = np.zeros_like(pos_x)
        
        # for label in temp.blink_time:
        #     start_blink = int(label-press_delay*250) - start
        #     end_blink = int(label-(press_delay-blink_period)*250) - start
        #     # print(f"{i}: {temp.blink_time}")
        #     # print(start_blink, end_blink)
        #     blink_label[start_blink:end_blink] = 1
    
        vl, vr, hl, hr = std['vl'][start:end], std['vr'][start:end], std['hl'][start:end], std['hr'][start:end]
        value_vl, value_vr, value_hl, value_hr = value['vl'][start:end], value['vr'][start:end], value['hl'][start:end], value['hr'][start:end]
        frame_std = np.stack([vl, vr, hl, hr, pos_x, pos_y, blink_label], axis=-1)
        frame_value = np.stack([value_vl, value_vr, value_hl, value_hr, pos_x, pos_y, blink_label], axis=-1)
        # print(frame.shape)
        df_std = pd.DataFrame(frame_std)
        df_std.rename(columns={0:'vl', 1:'vr', 2:'hl', 3:'hr', 4:'pos_x', 5:'pos_y', 6:'blink_label'}, inplace=True)
        df_value = pd.DataFrame(frame_value)
        df_value.rename(columns={0:'vl', 1:'vr', 2:'hl', 3:'hr', 4:'pos_x', 5:'pos_y', 6:'blink_label'}, inplace=True)
        result_std.append(df_std)
        result_value.append(df_value)

    result_std = pd.concat(result_std)
    result_value = pd.concat(result_value)
    return result_std, result_value

def peaks_intep(std):
    std = std.values
    delete_idx = []
    original_idx = np.arange(len(std))
    
    def peaks_finding(std):
        std_vl_peaks_id, _ = find_peaks(std[:, 0], distance = sample_rate * min_blink_period, prominence=(0.04, 1))
        print(std_vl_peaks_id)
        return std_vl_peaks_id

    std_vl_peaks_id = peaks_finding(std)
    
    for i, peaks_id in enumerate(std_vl_peaks_id):
        start_blink = int(peaks_id - blink_period * sample_rate / 2)
        end_blink = int(peaks_id + blink_period * sample_rate / 2)
        delete_idx.append(np.arange(start_blink, np.minimum(end_blink, std.shape[0]-1)))
    delete_idx = np.concatenate(delete_idx)
    valid_idx = np.delete(original_idx, delete_idx)
    interp_func_vl = interp1d(valid_idx, std[valid_idx, 0], kind='linear')
    interp_func_vr = interp1d(valid_idx, std[valid_idx, 1], kind='linear')
    
    std_interpolated = std.copy()
    
    std_interpolated[delete_idx, 0] = interp_func_vl(delete_idx) 
    std_interpolated[delete_idx, 1] = interp_func_vr(delete_idx) 
    
    std_interpolated = pd.DataFrame(std_interpolated)
    std_interpolated.rename(columns={0:'vl', 1:'vr', 2:'hl', 3:'hr', 4:'pos_x', 5:'pos_y', 6:'blink_label'}, inplace=True)
    return std_interpolated, std_vl_peaks_id

if __name__ == "__main__":
    processed_std = []
    processed_value = []
    intered_std = []
    
    for j in range(10):
        data_std, data_value = process_data(f"{csv_path}/Device_{j}_Volts.csv", f"{pkl_path}/Eyedata_{j}.pkl", 0.3, 0.1)
        inter_std, _ = peaks_intep(data_std)
        
        processed_std.append(data_std)
        processed_value.append(data_value)
        intered_std.append(inter_std)
        
        processed_std[j].to_csv(f"./data/latest data/std/train_std{j}.csv", index=False)
        processed_value[j].to_csv(f"./data/latest data/value/train_value{j}.csv", index=False)
        intered_std[j].to_csv(f"./data/latest data/std/intered_train_std{j}.csv", index=False)
        
    df_std = pd.concat(processed_std)
    df_std.rename(columns={0:'vl', 1:'vr', 2:'hl', 3:'hr', 4:'pos_x', 5:'pos_y', 6:'blink_label'}, inplace=True)
    # print(df_std)
    df_std.to_csv("./data/latest data/std/train_std_small.csv", index=False)
    
    df_value = pd.concat(processed_value)
    df_value.rename(columns={0:'vl', 1:'vr', 2:'hl', 3:'hr', 4:'pos_x', 5:'pos_y', 6:'blink_label'}, inplace=True)
    # print(df_value)
    df_value.to_csv("./data/latest data/value/train_value_small.csv", index=False)
    
    df_inter = pd.concat(intered_std)
    df_inter.rename(columns={0:'vl', 1:'vr', 2:'hl', 3:'hr', 4:'pos_x', 5:'pos_y', 6:'blink_label'}, inplace=True)
    # print(df_inter)
    df_inter.to_csv("./data/latest data/std/intered_train_std_small.csv", index=False)
    
    processed_std = []
    processed_value = []
    intered_std = []
    
    for j in range(25, 30):
        data_std, data_value = process_data(f"{csv_path}/Device_{j}_Volts.csv", f"{pkl_path}/Eyedata_{j}.pkl", 0.3, 0.1)
        inter_std, _ = peaks_intep(data_std)
        
        processed_std.append(data_std)
        processed_value.append(data_value)
        intered_std.append(inter_std)
        
        processed_std[j-25].to_csv(f"./data/latest data/std/test_std{j-25}.csv", index=False)
        processed_value[j-25].to_csv(f"./data/latest data/value/test_value{j-25}.csv", index=False)
        intered_std[j-25].to_csv(f"./data/latest data/std/intered_test_std{j-25}.csv", index=False)
        
    # df = pd.concat(processed)
    # df.rename(columns={0:'vl', 1:'vr', 2:'hl', 3:'hr', 4:'pos_x', 5:'pos_y', 6:'blink_label'}, inplace=True)
    # print(df)
    # df.to_csv("./data/latest data/continuous_test.csv", index=False)